import Combine
import Foundation
import SwiftUI

/// The Swift main-process's single source of truth for "is the local
/// Python backend reachable yet?"
///
/// Owns the subprocess handle (``PythonProcess``) and exposes the
/// typed HTTP client (``RubickClient``) once `/healthz` responds.
/// SwiftUI views observe ``status`` to enable / disable the search
/// box, render status pills, and decide which empty state to show.
@MainActor
final class BackendController: ObservableObject {
    @Published private(set) var status: BackendStatus = .idle
    @Published private(set) var client: RubickClient?

    /// Most recently fetched ``GET /index/stats`` snapshot, or
    /// ``nil`` before the first fetch / after a fetch failure. The
    /// status bar reads this for the live "N items · M chunks"
    /// counter; the Settings → Index tab still has its own refresh
    /// loop, but reading from here means the two surfaces converge
    /// after every ingest job finishes.
    ///
    /// Refresh policy (cheap on the backend — one ``count_rows`` per
    /// modality + a single ``to_pandas`` for distinct-doc counts):
    /// - Once when the backend reaches ``.ready``.
    /// - Once each time ``WatchService.activeJobIds`` transitions
    ///   from non-empty back to empty (i.e. every ingest job
    ///   completion). No background polling.
    @Published private(set) var indexStats: IndexStats?

    /// Most recently fetched ``GET /index/status`` snapshot — drives
    /// the Settings → Index "Pause / Resume" toggle + the status-bar
    /// indicator that turns the live spinner into a static pause icon
    /// when the user has parked ingest.
    ///
    /// Refresh policy mirrors ``indexStats``: once on backend ready,
    /// once each ingest-job drain edge, plus immediately after the
    /// user clicks Pause / Resume in Settings (the toggle owns its
    /// own optimistic round-trip and re-publishes through this
    /// snapshot). ``nil`` before the first fetch — UI hides any
    /// pause-derived chrome until the truth is known.
    @Published private(set) var indexQueueStatus: IndexQueueStatus?

    /// Watch service is shared at the app level so SwiftUI views can
    /// inspect / mutate the watched-folders list without depending on
    /// the controller's full surface. Lives on the controller so its
    /// lifetime is tied to the app's backend boot cycle.
    let watchService: WatchService

    /// Nebula window controller. Initialized after self is fully set up.
    private(set) var nebulaController: NebulaWindowController!

    private let process = PythonProcess()

    /// Combine subscription that fires ``refreshIndexStats()`` every
    /// time the watch service drains its in-flight job set. Held so
    /// the sink lives as long as the controller does.
    private var jobsDrainCancellable: AnyCancellable?

    /// Set to ``true`` while the watch service has at least one ingest
    /// job in flight. Used to detect the non-empty → empty edge that
    /// triggers a stats refresh (a `dropFirst` + `removeDuplicates`
    /// chain on Combine would be equivalent, but keeping the latch
    /// explicit makes the "refresh exactly once on drain" intent
    /// readable).
    private var jobsWereActive = false

    init(watchService: WatchService? = nil) {
        // Default: fresh folders store reads from ``UserDefaults``.
        // Tests can inject a stubbed service via the parameter.
        self.watchService = watchService ?? WatchService(folders: WatchedFoldersStore())
        // Two-phase: nebulaController needs `self` reference
        self.nebulaController = nil
        self.nebulaController = NebulaWindowController(backendController: self)
    }

    /// Total wall-clock budget for boot (port discovery + uvicorn
    /// import + first `/healthz` success). The Python side is very
    /// fast at startup (modules are lazy-imported), so 30 s is huge
    /// headroom — on a clean machine the actual time is ~2-3 s.
    private let bootTimeout: TimeInterval = 30

    /// Poll interval for `/healthz` while waiting for boot. Short
    /// enough to feel snappy in the UI, long enough that we don't
    /// spam the event loop in a tight loop.
    private let pollInterval: UInt64 = 200_000_000  // 200 ms in ns

    func start() async {
        guard case .idle = status else { return }
        status = .starting
        do {
            let runtime = BackendRuntime.resolve()
            let port = try PythonProcess.findFreePort()
            try process.launch(runtime: runtime, port: port)
            let candidateClient = RubickClient(port: port)
            try await waitUntilHealthy(client: candidateClient)
            self.client = candidateClient
            self.status = .ready(port: port)
            // Backend is up — tell the watch service so it can run an
            // initial scan of any persisted watched folders and start
            // forwarding FSEvents-triggered jobs.
            watchService.bind(client: candidateClient)
            // Status-bar live counter.
            // First fetch happens unconditionally so the counter lands
            // populated for users with an existing index. Subsequent
            // fetches are wired up to the watch service so every
            // job-drain edge refreshes the snapshot. Both run in
            // detached tasks so they never delay backend-ready ↔ UI.
            Task { await self.refreshIndexStats() }
            Task { await self.refreshIndexQueueStatus() }
            installJobsDrainHook()
        } catch {
            self.status = .failed(reason: error.localizedDescription)
            process.terminate()
        }
    }

    func shutdown() {
        process.terminate()
        status = .idle
        client = nil
        indexStats = nil
        indexQueueStatus = nil
        jobsDrainCancellable?.cancel()
        jobsDrainCancellable = nil
        jobsWereActive = false
        watchService.bind(client: nil)
    }

    // MARK: - Index stats

    /// Re-fetch ``GET /index/stats`` and publish the result. No-op
    /// (and a benign error log) if the backend isn't connected yet.
    /// Cheap to call repeatedly — the backend wraps the underlying
    /// ``count_rows`` / ``to_pandas`` in ``asyncio.to_thread`` so
    /// this never blocks an in-flight ``/search`` query.
    func refreshIndexStats() async {
        guard let client else { return }
        do {
            let fresh = try await client.indexStats()
            self.indexStats = fresh
        } catch {
            // Don't clobber a previously-good snapshot on a transient
            // error; the next drain edge will retry. Just log so dev
            // builds can see persistent failures.
            #if DEBUG
            print("BackendController.refreshIndexStats failed: \(error)")
            #endif
        }
    }

    /// Re-fetch ``GET /index/status`` and publish the result. Same
    /// "don't clobber on transient error" policy as ``refreshIndexStats``
    /// — pause / resume state is sticky on the backend, so a missed
    /// refresh is harmless until the next drain edge catches up.
    func refreshIndexQueueStatus() async {
        guard let client else { return }
        do {
            let fresh = try await client.indexQueueStatus()
            self.indexQueueStatus = fresh
        } catch {
            #if DEBUG
            print("BackendController.refreshIndexQueueStatus failed: \(error)")
            #endif
        }
    }

    /// Toggle the backend's ingest dispatch gate. Owns the optimistic
    /// re-publish on success so SwiftUI buttons can ``await`` this
    /// without a separate ``refresh`` call.
    ///
    /// Errors are bubbled up so the call site (currently the
    /// Settings → Index toggle) can surface them inline rather than
    /// being silently swallowed.
    func setIndexPaused(_ paused: Bool) async throws {
        guard let client else {
            throw RubickClientError.transport(
                "Backend not connected (status: \(status.label))."
            )
        }
        let fresh = paused
            ? try await client.pauseIndex()
            : try await client.resumeIndex()
        self.indexQueueStatus = fresh
    }

    /// Subscribe to ``WatchService.activeJobIds`` and refresh stats
    /// every time the set transitions from non-empty back to empty.
    /// "Every job has drained" is the only point where the LanceDB
    /// row counts can change in normal operation, so polling outside
    /// of that edge would just burn CPU for no observable benefit.
    private func installJobsDrainHook() {
        jobsDrainCancellable?.cancel()
        jobsWereActive = !watchService.activeJobIds.isEmpty
        jobsDrainCancellable = watchService.$activeJobIds
            .receive(on: RunLoop.main)
            .sink { [weak self] ids in
                guard let self else { return }
                let active = !ids.isEmpty
                if self.jobsWereActive && !active {
                    Task { await self.refreshIndexStats() }
                    Task { await self.refreshIndexQueueStatus() }
                    // Auto-recompute nebula map after all ingest jobs finish
                    Task {
                        try? await self.client?.nebulaRecompute()
                    }
                }
                self.jobsWereActive = active
            }
    }

    // MARK: - Health polling

    private func waitUntilHealthy(client: RubickClient) async throws {
        let deadline = Date().addingTimeInterval(bootTimeout)
        var lastError: Error?
        while Date() < deadline {
            do {
                let h = try await client.healthz()
                if h.status == "ok" { return }
            } catch {
                lastError = error
            }
            try? await Task.sleep(nanoseconds: pollInterval)
        }
        throw PythonProcessError.healthCheckTimedOut(seconds: bootTimeout)
            .with(cause: lastError)
    }
}

// MARK: - Status

enum BackendStatus: Equatable {
    case idle
    case starting
    case ready(port: UInt16)
    case failed(reason: String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var label: String {
        switch self {
        case .idle: return "Idle"
        case .starting: return "Starting local backend…"
        case .ready(let p): return "Connected · 127.0.0.1:\(p)"
        case .failed(let r): return "Failed — \(r)"
        }
    }

    var color: Color {
        switch self {
        case .idle: return .gray
        case .starting: return .yellow
        case .ready: return .green
        case .failed: return .red
        }
    }
}

// MARK: - Error chaining

private extension PythonProcessError {
    /// Tack the most recent transport error onto the timeout message so
    /// the UI shows e.g. "…within 30s. Last error: Connection refused"
    /// instead of an unhelpful bare timeout.
    func with(cause: Error?) -> Error {
        guard let cause else { return self }
        switch self {
        case .healthCheckTimedOut(let seconds):
            let composed =
                "Backend did not become healthy within \(Int(seconds))s. "
                + "Last error: \(cause.localizedDescription)"
            return NSError(
                domain: "Rubick.PythonProcess",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: composed]
            )
        default:
            return self
        }
    }
}
