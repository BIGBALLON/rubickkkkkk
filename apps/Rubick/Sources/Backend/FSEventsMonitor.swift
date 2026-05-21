import Foundation

/// Thin Swift wrapper around the CoreServices `FSEvents` API.
///
/// macOS surfaces filesystem change notifications via a stream of
/// callbacks scoped to one-or-more watch paths. We use the C-level
/// ``FSEventStreamCreate`` + ``FSEventStreamScheduleWithRunLoop`` API
/// (rather than DispatchSource's per-file `.vnode` source) because:
///
/// - We watch *directory subtrees*, not individual files. FSEvents
///   is the only first-class macOS API for that — DispatchSource
///   would require manually scaling fd-per-file, which blows up on
///   any non-trivial folder.
/// - FSEvents coalesces "rapid changes to the same dir" into one
///   notification at the kernel level (the ``latency`` arg), which
///   keeps us off the wakeup-storm path during e.g. a `cp -r`.
///
/// Public API: ``start(paths:)`` and ``stop()``. The monitor
/// delivers batched changes via ``onBatch`` after a small additional
/// **userspace debounce** (default 1.5 s of quiet) — necessary because
/// FSEvents' built-in `latency` only coalesces *within* one wake-up,
/// not across them. A `cp 100files/` shows up as e.g. 4 callbacks
/// 100-300 ms apart, and we want all 100 paths in a single
/// downstream POST.
///
/// Thread model: callbacks fire on the main run loop (we schedule
/// on `RunLoop.main`), and the debounce timer is also main-actor —
/// so callers' ``onBatch`` closure runs on the main thread without
/// extra locking. The monitor is ``@MainActor`` because of this.
@MainActor
final class FSEventsMonitor {
    /// Invoked when the userspace debounce window expires after one
    /// or more FSEvents callbacks. ``urls`` is deduplicated and may
    /// contain a mix of file URLs and directory URLs depending on
    /// what changed.
    var onBatch: (@MainActor ([URL]) -> Void)?

    /// Apple recommends 0.5–2 s kernel latency; 0.5 keeps re-ingest snappy
    /// after save.
    private let kernelLatency: CFTimeInterval = 0.5

    /// Userspace quiet-period before we flush a batch. Sized so that
    /// a typical `cp -r` of a 1000-file folder lands in one batch.
    private let userspaceDebounce: TimeInterval = 1.5

    /// ``nonisolated(unsafe)`` because the FSEventStream API is
    /// documented (CoreServices/Reference) as thread-safe — stop /
    /// invalidate / release work from any thread. The pointer
    /// itself is a typedef of ``ConstFSEventStreamRef`` (OpaquePointer),
    /// which Swift can't prove Sendable. We confine writes to the
    /// MainActor and let deinit read it from whatever queue.
    nonisolated(unsafe) private var stream: FSEventStreamRef?
    private var pending: Set<String> = []
    private var debounceTimer: Timer?
    private let dispatchQueue = DispatchQueue.main

    deinit {
        // FSEventStream cleanup is thread-safe; release whatever the
        // last MainActor write left in ``stream``. Calling stop on a
        // non-running stream is a no-op.
        if let s = stream {
            FSEventStreamStop(s)
            FSEventStreamInvalidate(s)
            FSEventStreamRelease(s)
        }
    }

    // MARK: - Lifecycle

    /// Begin watching ``paths`` recursively. Replaces any previous
    /// watch — calling ``start`` twice doesn't merge sets.
    ///
    /// No-op (and ``stop()``-equivalent) when ``paths`` is empty —
    /// FSEvents rejects empty path arrays with ``__FSEventStreamRefInvalid``.
    func start(paths: [URL]) {
        stop()
        guard !paths.isEmpty else { return }

        let pathStrings = paths.map(\.path) as CFArray

        // We use ``kFSEventStreamEventIdSinceNow`` rather than a
        // persisted ``lastEventId`` because Rubick's source of
        // truth is what's in LanceDB — we don't replay missed events,
        // we just re-scan on next startup via the watched-folders
        // store. Persisted event IDs would let us play catch-up
        // across restarts but cost a notable amount of complexity
        // (handling "FSEvents history was rotated" → full re-scan).
        var context = createCallbackContext()

        let flags: FSEventStreamCreateFlags =
            UInt32(kFSEventStreamCreateFlagFileEvents)
            | UInt32(kFSEventStreamCreateFlagUseCFTypes)
            | UInt32(kFSEventStreamCreateFlagNoDefer)
            | UInt32(kFSEventStreamCreateFlagWatchRoot)

        let createdStream: FSEventStreamRef? = withUnsafeMutablePointer(to: &context) { ctxPtr in
            FSEventStreamCreate(
                kCFAllocatorDefault,
                Self.streamCallback,
                ctxPtr,
                pathStrings,
                FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
                kernelLatency,
                flags
            )
        }
        guard let s = createdStream else {
            FileHandle.standardError.write(
                Data("[FSEventsMonitor] FSEventStreamCreate failed\n".utf8)
            )
            return
        }
        // ``SetDispatchQueue`` replaces the macOS-13-deprecated
        // ``ScheduleWithRunLoop``; we still use the main queue so
        // MainActor.assumeIsolated in the callback stays sound.
        FSEventStreamSetDispatchQueue(s, dispatchQueue)
        if !FSEventStreamStart(s) {
            FSEventStreamInvalidate(s)
            FSEventStreamRelease(s)
            FileHandle.standardError.write(
                Data("[FSEventsMonitor] FSEventStreamStart failed\n".utf8)
            )
            return
        }
        stream = s
    }

    func stop() {
        debounceTimer?.invalidate()
        debounceTimer = nil
        pending.removeAll()

        guard let s = stream else { return }
        FSEventStreamStop(s)
        FSEventStreamInvalidate(s)
        FSEventStreamRelease(s)
        stream = nil
    }

    // MARK: - Callback plumbing

    /// We pass a long-lived pointer-to-self into the FSEventStream's
    /// ``info`` slot so the C callback can route events back to us.
    /// ``Unmanaged.passUnretained`` is correct here because the
    /// monitor outlives the stream (we ``stop()`` first in ``deinit``).
    private func createCallbackContext() -> FSEventStreamContext {
        let ptr = Unmanaged.passUnretained(self).toOpaque()
        return FSEventStreamContext(
            version: 0,
            info: ptr,
            retain: nil,
            release: nil,
            copyDescription: nil
        )
    }

    /// C-style callback (``@convention(c)``). Routed back to the
    /// originating ``FSEventsMonitor`` via the ``info`` slot we
    /// stashed in :func:`createCallbackContext`. Must be a
    /// non-capturing static so it decays to a function pointer.
    private static let streamCallback: FSEventStreamCallback = {
        _, info, count, paths, flags, _ in
        guard let info else { return }
        let monitor = Unmanaged<FSEventsMonitor>.fromOpaque(info).takeUnretainedValue()
        // ``kFSEventStreamCreateFlagUseCFTypes`` makes paths a CFArray
        // of CFStrings — bridge straight to ``[String]``.
        let cfArray = Unmanaged<CFArray>.fromOpaque(paths).takeUnretainedValue()
        guard let array = cfArray as? [String] else { return }
        let flagsBuffer = UnsafeBufferPointer(start: flags, count: count)
        let flagsCopy = Array(flagsBuffer)
        // FSEvents already calls us on the main run loop (we
        // scheduled it there), but ``MainActor`` isolation can't be
        // proved by the compiler across the C callback boundary.
        // ``assumeIsolated`` is sound here because of the scheduling.
        MainActor.assumeIsolated {
            monitor.handleEvents(paths: array, flags: flagsCopy)
        }
    }

    private func handleEvents(paths: [String], flags: [FSEventStreamEventFlags]) {
        for (i, raw) in paths.enumerated() {
            // Drop events we never want to re-ingest on. ``History
            // Done`` arrives once after start; ``Mount`` / ``Unmount``
            // is volume attach/detach noise; ``RootChanged`` is the
            // watch root itself moving (we don't try to follow).
            let flag = flags[i]
            let ignore =
                (flag & UInt32(kFSEventStreamEventFlagHistoryDone)) != 0
                || (flag & UInt32(kFSEventStreamEventFlagMount)) != 0
                || (flag & UInt32(kFSEventStreamEventFlagUnmount)) != 0
                || (flag & UInt32(kFSEventStreamEventFlagRootChanged)) != 0
            if ignore { continue }
            pending.insert(raw)
        }
        scheduleFlush()
    }

    private func scheduleFlush() {
        debounceTimer?.invalidate()
        debounceTimer = Timer.scheduledTimer(
            withTimeInterval: userspaceDebounce,
            repeats: false
        ) { [weak self] _ in
            // Timer callbacks fire on the run loop they were
            // scheduled on (main) and are non-isolated. Hop back
            // explicitly so ``flush()`` keeps its actor isolation.
            Task { @MainActor [weak self] in
                self?.flush()
            }
        }
    }

    private func flush() {
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        if batch.isEmpty { return }
        let urls = batch.map { URL(fileURLWithPath: $0) }.sorted { $0.path < $1.path }
        onBatch?(urls)
    }
}
