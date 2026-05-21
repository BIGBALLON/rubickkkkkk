import Combine
import Foundation

/// Glue between :class:`WatchedFoldersStore`, :class:`FSEventsMonitor`,
/// and :class:`RubickClient`.
///
/// Responsibilities:
///
/// 1. Re-arm the FSEventsMonitor whenever the user adds / removes a
///    watched folder, so we never index folders the user no longer
///    cares about.
/// 2. Drive an **initial scan** when a folder is freshly added — the
///    FSEventsMonitor only fires on *future* changes, so without
///    this the contents of an existing folder would never be seen.
/// 3. Convert FSEvents callbacks into ``POST /index/job`` calls,
///    then poll status and write it back to the
///    ``WatchedFoldersStore`` for UI consumption.
/// 4. Survive backend restarts: when the controller flips
///    ``client`` to a new instance after a re-spawn, we rebind.
///
/// FSEvents debounce (1.5 s) coalesces rapid writes; the
/// backend path+mtime / content dedup silently absorbs the duplicates a
/// single Save can produce — so this layer can be straightforward
/// "events → POST".
@MainActor
final class WatchService: ObservableObject {
    @Published private(set) var activeJobIds: Set<String> = []

    /// ``true`` while ``refreshAll()`` is iterating its per-folder
    /// kick loop. Drives the spinning animation on the sidebar's
    /// refresh button and gates the button so the user
    /// can't queue a second rescan on top of a running one.
    @Published private(set) var isRefreshing = false

    /// User-controlled re-scan policy. Defaults to
    /// ``.live`` — current behaviour for users who haven't touched
    /// the new Settings → General → "Watch mode" panel.
    /// Persisted to ``UserDefaults`` (``watch_mode_*`` keys); see
    /// ``WatchModePersistence`` for the (de)serialisation surface.
    @Published var watchMode: WatchMode = .live {
        didSet {
            guard watchMode != oldValue else { return }
            WatchModePersistence.save(watchMode, defaults: defaults)
            applyWatchMode()
        }
    }

    let folders: WatchedFoldersStore
    private let monitor = FSEventsMonitor()
    private var client: RubickClient?
    private var foldersSink: AnyCancellable?
    /// Combine cancellable for the periodic-rescan timer used by
    /// ``WatchMode.scheduled``. Held so we can tear it down cleanly
    /// when the user flips back to ``.live`` / ``.manual``.
    private var scheduledRescanTimer: AnyCancellable?
    /// UserDefaults instance — injectable so unit tests can drive
    /// the persistence path without touching the real plist.
    private let defaults: UserDefaults

    init(
        folders: WatchedFoldersStore,
        defaults: UserDefaults = .standard
    ) {
        self.folders = folders
        self.defaults = defaults
        self.watchMode = WatchModePersistence.load(defaults: defaults)
        // Re-arm the monitor whenever the watch list changes.
        foldersSink = folders.$folders
            .removeDuplicates { lhs, rhs in
                lhs.map(\.id) == rhs.map(\.id)
            }
            .sink { [weak self] list in
                Task { @MainActor in
                    self?.rearmMonitor(list: list)
                }
            }
        monitor.onBatch = { [weak self] urls in
            self?.handleBatch(urls)
        }
        // Apply the persisted mode once the timer / monitor wires
        // are in place. ``.live`` / ``.manual`` take effect lazily
        // (next FSEvents callback consults ``watchMode``);
        // ``.scheduled`` arms the timer here.
        applyWatchMode()
    }

    // MARK: - Public API used by BackendController / Nebula

    /// Inform the service about the live backend client. Called by
    /// ``BackendController`` once ``/healthz`` answers OK.
    func bind(client: RubickClient?) {
        self.client = client
        // Backend just came up — give it the current watch list as
        // an initial scan (per folder, async, fire-and-forget). The
        // scan re-walks each folder; backend dedup + path-cache
        // make unchanged-bytes files essentially free, so the cost
        // is bounded by "files whose bytes actually changed since
        // the last launch". The kick() return path also queries the
        // live total per folder so the sidebar's
        // ``items · chunks`` line lands populated on first launch
        // (the in-memory stats from the previous session are gone).
        if client != nil {
            for folder in folders.folders {
                Task { await self.kick(folder, paths: [folder.url]) }
            }
        }
    }

    /// Add ``url`` to the watch list and immediately scan it. The
    /// returned bool mirrors ``WatchedFoldersStore.add`` (false ==
    /// duplicate).
    @discardableResult
    func addFolder(_ url: URL) -> Bool {
        let added = folders.add(url)
        if added, let folder = folders.folders.first(where: { $0.url == url }) {
            Task { await self.kick(folder, paths: [url]) }
        }
        return added
    }

    func removeFolder(_ folder: WatchedFolder) {
        folders.remove(folder)
    }

    /// Re-scan every watched folder from the top.
    ///
    /// Runs every per-folder kick **sequentially** so the backend's
    /// priority queue isn't flooded with N parallel index jobs (a
    /// hot search would queue up behind every one of them).
    /// ``isRefreshing`` flips to ``true`` while the loop runs so
    /// the sidebar refresh button can render its spinner; it flips
    /// back to ``false`` even on partial failure (any per-folder
    /// failure is logged into ``WatchedFolder.status`` by the
    /// existing ``kick`` path, so we don't need a separate
    /// reporting channel here).
    ///
    /// Backend dedup (``doc_id = sha256(file_bytes)``) makes
    /// re-kicks essentially free on the unchanged-files majority —
    /// only files whose bytes actually changed pay the embed cost.
    /// That's also what makes the scheduled-rescan mode safe to
    /// wire on a 60-min timer.
    func refreshAll() async {
        guard !isRefreshing, client != nil else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        for folder in folders.folders {
            await kick(folder, paths: [folder.url])
        }
    }

    /// Re-scan exactly one folder (right-click context-menu entry).
    /// Doesn't toggle ``isRefreshing`` (the global refresh button is
    /// the only path that owns that flag) — a single-folder rescan
    /// is a quick targeted operation, not a "wait for the whole
    /// world" gesture.
    func refreshFolder(_ folder: WatchedFolder) async {
        guard client != nil else { return }
        await kick(folder, paths: [folder.url])
    }

    // MARK: - Internal plumbing

    private func rearmMonitor(list: [WatchedFolder]) {
        monitor.start(paths: list.map(\.url))
    }

    private func handleBatch(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        // Respect the user's watch-mode choice. ``.manual``
        // keeps the FSEvents stream up (so a future flip back to
        // ``.live`` doesn't need a re-arm) but drops the auto-kick;
        // the user re-scans on demand via the sidebar refresh button.
        guard watchMode.kicksOnFSEvents else { return }
        // Group changed paths under whichever watched folder they
        // belong to, so each backend job is per-folder (cleaner UI).
        var byFolder: [WatchedFolder: [URL]] = [:]
        for url in urls {
            guard let folder = matchingFolder(for: url) else { continue }
            byFolder[folder, default: []].append(url)
        }
        for (folder, paths) in byFolder {
            Task { await self.kick(folder, paths: paths) }
        }
    }

    /// (Re-)install the periodic-rescan timer iff the current mode
    /// asks for one. ``.live`` and ``.manual`` both tear the timer
    /// down — ``.live`` doesn't need it (FSEvents is already
    /// triggering kicks) and ``.manual`` shouldn't trigger any
    /// background work at all.
    ///
    /// Idempotent: every flip to ``watchMode`` calls through here,
    /// and the first action is to cancel any in-flight subscription
    /// so we never accumulate timers.
    private func applyWatchMode() {
        scheduledRescanTimer?.cancel()
        scheduledRescanTimer = nil
        guard case .scheduled(let intervalMinutes) = watchMode else { return }
        let seconds = TimeInterval(max(1, intervalMinutes) * 60)
        // ``Timer.publish`` on the main RunLoop + ``.autoconnect``
        // means we don't have to manage start/stop manually beyond
        // holding the cancellable. Sink is on the main actor (the
        // service is @MainActor) so calling refreshAll() directly
        // is safe.
        scheduledRescanTimer = Timer.publish(
            every: seconds,
            on: .main,
            in: .common
        )
        .autoconnect()
        .sink { [weak self] _ in
            Task { @MainActor in
                await self?.refreshAll()
            }
        }
    }

    /// Find the watched folder that ``url`` lives inside. Falls back
    /// to ``nil`` (drop the event) if no folder is an ancestor —
    /// shouldn't happen since FSEvents only fires for paths under
    /// our watch list, but be defensive.
    private func matchingFolder(for url: URL) -> WatchedFolder? {
        let target = url.standardized.path
        return folders.folders.first { folder in
            let prefix = folder.url.standardized.path
            return target == prefix || target.hasPrefix(prefix + "/")
        }
    }

    /// Send an ingest job for ``paths`` (already known to belong to
    /// ``folder``), then poll until terminal status and write the
    /// outcome back to the store so the UI reflects it.
    ///
    /// After a successful job we issue a second round-trip to
    /// ``GET /index/stats?path_prefix=…`` so the
    /// ``ready(at:, stats:)`` we publish carries the **total** count
    /// of items / chunks under this folder, not just the
    /// incremental "what did this job add" delta. This matters for
    /// dedup-heavy re-scans (cold start, scheduled rescan): without
    /// it, every "0 new files" outcome would show ``0 items``
    /// instead of the real totals.
    private func kick(_ folder: WatchedFolder, paths: [URL]) async {
        guard let client else { return }

        do {
            let job = try await client.enqueueIndexJob(paths: paths)
            activeJobIds.insert(job.id)
            folders.setStatus(
                .indexing(jobId: job.id, progress: nil),
                for: folder
            )

            // Push progress back to the store on every poll so the
            // sidebar card + IngestProgressBanner can render a real
            // ``ProgressView(value:total:)`` instead of a spinner.
            // ``onProgress`` runs on the polling task; hop to MainActor
            // before mutating ``WatchedFoldersStore`` (it's
            // ``@MainActor``-isolated).
            let folderId = folder.id
            let final = try await client.waitForIndexJob(
                id: job.id,
                onProgress: { [weak self] update in
                    Task { @MainActor in
                        guard let self else { return }
                        guard
                            let live = self.folders.folders.first(
                                where: { $0.id == folderId }
                            )
                        else { return }
                        // Only update while the status is still
                        // ``.indexing(jobId: job.id, ...)`` — we don't
                        // want a late poll to overwrite a terminal
                        // ``.ready`` / ``.failed`` set below.
                        if case .indexing(let activeJobId, _) = live.status,
                           activeJobId == update.id {
                            self.folders.setStatus(
                                .indexing(jobId: update.id, progress: update.progress),
                                for: live
                            )
                        }
                    }
                }
            )
            activeJobIds.remove(job.id)
            switch final.status {
            case .succeeded:
                let totals = await fetchFolderTotals(folder: folder)
                folders.setStatus(
                    .ready(at: Date(), stats: totals ?? final.stats
                        ?? .init(files: 0, chunks: 0, skipped: 0)),
                    for: folder
                )
            case .failed:
                folders.setStatus(.failed(reason: final.error ?? "unknown"), for: folder)
            case .queued, .running:
                // ``waitForIndexJob`` only returns on terminal status,
                // so this branch is unreachable — keep it explicit.
                folders.setStatus(.failed(reason: "unexpected status"), for: folder)
            }
        } catch {
            folders.setStatus(.failed(reason: error.localizedDescription), for: folder)
        }
    }

    /// Query ``GET /index/stats?path_prefix=<folder>`` and return an
    /// ``IndexJob.Stats`` shaped to fit the existing
    /// ``WatchedFolder.Status.ready(at:, stats:)`` payload. The
    /// stats record only has ``files`` / ``chunks`` / ``skipped`` —
    /// we map ``totalDocs`` → ``files`` and ``totalChunks`` →
    /// ``chunks`` because that's what the sidebar reads (and the
    /// rest of the IndexStats envelope is exposed elsewhere on
    /// ``BackendController``).
    ///
    /// Returns ``nil`` on transport error so the caller falls back
    /// to the incremental stats from the ingest job — better than
    /// blanking the sidebar on a flaky single request.
    private func fetchFolderTotals(folder: WatchedFolder) async -> IndexJob.Stats? {
        guard let client else { return nil }
        do {
            let snapshot = try await client.indexStats(
                pathPrefix: folder.url.standardized.path
            )
            return IndexJob.Stats(
                files: snapshot.totalDocs,
                chunks: snapshot.totalChunks,
                skipped: 0
            )
        } catch {
            #if DEBUG
            print("WatchService.fetchFolderTotals failed: \(error)")
            #endif
            return nil
        }
    }
}

// MARK: - Watch mode

/// User-controlled re-scan policy. Surfaced in Settings → General;
/// persisted to ``UserDefaults`` so the choice survives relaunch.
///
/// All three modes leave the FSEvents stream up (kernel-level cost
/// is essentially zero, and tearing it down would mean we couldn't
/// flip back to ``.live`` without re-arming on every folder); the
/// difference is purely in what ``handleBatch`` does with the
/// callbacks and whether the periodic timer is wired.
enum WatchMode: Equatable, Hashable, Sendable {
    /// FSEvents callback → immediate ``kick``. No timer. Best for
    /// users actively editing a watched folder who want every save
    /// to land in the index without delay; most users won't need
    /// this — it tends to flood the queue on multi-thousand-image
    /// folders.
    case live
    /// **Default.** Pure timer: ``refreshAll()`` every
    /// ``intervalMinutes`` minutes. FSEvents callbacks are
    /// ignored, so editing a watched folder doesn't queue an
    /// ingest job per save — the next periodic re-scan picks
    /// everything up. Switched from "live + safety net" to pure
    /// timer in the v1.x ingest cleanup; existing users who had
    /// ``.scheduled`` selected pick up the new semantics on next
    /// launch.
    case scheduled(intervalMinutes: Int)
    /// FSEvents stream stays up but its callbacks are ignored —
    /// the user must hit the sidebar refresh button to re-index.
    /// Useful for huge folders where every Save shouldn't trigger
    /// a re-scan (e.g. an active build directory).
    case manual

    /// Whether an FSEvents batch should fire ``kick``. Only
    /// ``.live`` returns ``true``; ``.scheduled`` and ``.manual``
    /// both ignore real-time events (``.scheduled`` re-scans on
    /// its timer, ``.manual`` waits for the user).
    var kicksOnFSEvents: Bool {
        switch self {
        case .live: return true
        case .scheduled, .manual: return false
        }
    }

    /// Stable short slug for ``UserDefaults`` (case-name only;
    /// ``.scheduled``'s payload is stored separately so the schema
    /// stays a single string + an int).
    var persistKey: String {
        switch self {
        case .live: return "live"
        case .scheduled: return "scheduled"
        case .manual: return "manual"
        }
    }

    /// Default interval for ``.scheduled`` mode (60 minutes).
    static let defaultScheduledIntervalMinutes: Int = 60

    /// Bounds for the Settings → General stepper. Below 5 min the
    /// LanceDB churn outweighs the safety-net benefit; above 24 h
    /// we may as well tell the user to use ``.manual``.
    static let scheduledIntervalRange: ClosedRange<Int> = 5...(24 * 60)
}

/// Tiny serialisation helper kept out of ``WatchService`` so the
/// service body stays focused on FSEvents + ingest plumbing.
/// Stores two ``UserDefaults`` keys:
///
/// - ``watch_mode`` — case slug (``"live"`` / ``"scheduled"`` /
///   ``"manual"``)
/// - ``watch_mode_interval_minutes`` — integer payload, only
///   meaningful for ``.scheduled``
///
/// Keeping the schema flat (string + int) instead of one
/// JSON-encoded blob means a hand-edit of the plist still works,
/// and adding a future fourth mode doesn't risk wedging the
/// existing decoder.
enum WatchModePersistence {
    static let modeKey = "watch_mode"
    static let intervalKey = "watch_mode_interval_minutes"

    static func load(defaults: UserDefaults) -> WatchMode {
        // Default — for users who haven't touched Settings → General —
        // is now ``.scheduled(60min)`` rather than ``.live``. Real-time
        // FSEvents kicks ended up flooding the queue on multi-thousand-
        // image folders, so we make pure-timer the polite default and
        // let high-cadence users opt into ``.live``.
        guard let raw = defaults.string(forKey: modeKey) else {
            return .scheduled(intervalMinutes: WatchMode.defaultScheduledIntervalMinutes)
        }
        switch raw {
        case "live":
            return .live
        case "manual":
            return .manual
        case "scheduled":
            let stored = defaults.integer(forKey: intervalKey)
            let clamped = stored == 0
                ? WatchMode.defaultScheduledIntervalMinutes
                : stored
            return .scheduled(intervalMinutes: clamped)
        default:
            // Unknown / corrupted value — fall back to the safe
            // default so a bad write doesn't lock the user out.
            // We deliberately don't raise; the next legitimate write
            // will overwrite the bad value.
            return .scheduled(intervalMinutes: WatchMode.defaultScheduledIntervalMinutes)
        }
    }

    static func save(_ mode: WatchMode, defaults: UserDefaults) {
        defaults.set(mode.persistKey, forKey: modeKey)
        if case .scheduled(let m) = mode {
            defaults.set(m, forKey: intervalKey)
        }
        // Intentionally don't clear the interval key when leaving
        // ``.scheduled``: keeping it around means flipping back
        // restores the user's chosen interval instead of dropping
        // them to the default.
    }
}
