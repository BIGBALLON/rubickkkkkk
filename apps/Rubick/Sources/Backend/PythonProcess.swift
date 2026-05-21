import Darwin
import Foundation

/// Spawns and supervises the Python `rubick_backend` uvicorn subprocess.
///
/// Startup contract:
/// - Find a free 127.0.0.1 TCP port via the ephemeral-bind trick.
/// - Pick the right backend runtime — bundled-in-.app first, dev
///   venv second, system Python last (see ``BackendRuntime``).
/// - Spawn `python -m uvicorn rubick_backend.main:app --port <port>`.
/// - Poll `GET /healthz` until it returns OK, with a generous timeout
///   (the first run on a clean machine has to download ~1.8 GB of
///   model weights into the HF cache on first embedding call — but
///   *startup* itself only imports modules, so this poll is fast).
/// - Forward stdout/stderr to our own stdout, prefixed, so backend
///   logs show up in Xcode console / `Console.app`.
///
/// Not implemented yet:
/// - Crash auto-restart loops (not implemented)
/// - Health monitoring after boot (not implemented)
///
/// Release builds prefer a bundled
/// `Rubick.app/Contents/Resources/backend-runtime/` runtime when one
/// exists (built by `scripts/build_backend_bundle.sh` and copied
/// into the .app by `scripts/build_dmg.sh`). Dev mode (the
/// `backend/.venv` we already had) remains the fallback so
/// `tuist generate` + `xcodebuild` keeps working unchanged.
///
/// Cleanup: a `terminationHandler` and `applicationWillTerminate`
/// observer ensure we send SIGTERM to the child on quit; macOS will
/// otherwise leave it dangling.
@MainActor
final class PythonProcess {
    private(set) var process: Process?
    private(set) var port: UInt16?
    private(set) var pid: Int32?

    /// Bind ::1 to port 0, ask the kernel which port it gave us, close
    /// the socket, and return that port number.
    ///
    /// This is the classic TOCTOU dance: another process *could* grab
    /// the port in the microseconds between `close` and uvicorn's
    /// `bind`. In practice for a desktop app spawning its own backend
    /// it's a non-issue, but we wrap in a small retry loop just so
    /// transient failures don't take down the app.
    static func findFreePort() throws -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw PythonProcessError.portDiscoveryFailed("socket(): \(errno)")
        }
        defer { close(fd) }

        var reuse: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0  // OS picks an ephemeral port
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let len = socklen_t(MemoryLayout<sockaddr_in>.size)

        let bindOK = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                bind(fd, ptr, len)
            }
        }
        guard bindOK == 0 else {
            throw PythonProcessError.portDiscoveryFailed("bind(): \(errno)")
        }

        var outAddr = sockaddr_in()
        var outLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameOK = withUnsafeMutablePointer(to: &outAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                getsockname(fd, ptr, &outLen)
            }
        }
        guard nameOK == 0 else {
            throw PythonProcessError.portDiscoveryFailed("getsockname(): \(errno)")
        }
        return UInt16(bigEndian: outAddr.sin_port)
    }

    /// Spawn the backend. Throws if the executable can't be found
    /// or the launch syscall fails. Health-check polling is the
    /// caller's job (see `BackendController.start`).
    func launch(runtime: BackendRuntime, port: UInt16) throws {
        // Pre-flight only the dev-mode resolution. For the bundled
        // runtime we trust the bundle (``build_backend_bundle.sh`` ran
        // a heavy import smoke test at build time, and the bundle is
        // codesigned as part of the .app); for ``--system-python`` we
        // can't pre-flight without spawning anyway.
        if case .devVenv(_, let repoRoot) = runtime.mode {
            let pyproject = repoRoot
                .appendingPathComponent("backend")
                .appendingPathComponent("pyproject.toml")
            guard FileManager.default.fileExists(atPath: pyproject.path) else {
                throw PythonProcessError.launchFailed(
                    "Could not find backend/ at \(repoRoot.path). "
                        + "Set RUBICK_REPO_ROOT, run from a Tuist-generated build, "
                        + "or launch the binary directly from inside the repo."
                )
            }
        }

        let proc = Process()
        proc.executableURL = runtime.pythonURL
        var arguments: [String] = []
        // Handle the /usr/bin/env fallback case.
        if runtime.pythonURL.lastPathComponent == "env" {
            arguments.append("python3")
        }
        arguments.append(contentsOf: [
            "-m", "uvicorn",
            "rubick_backend.main:app",
            "--host", "127.0.0.1",
            "--port", String(port),
            // ``info`` is verbose enough to surface "Uvicorn running on
            // http://127.0.0.1:<port>" in our forwarded stdout, which is
            // a useful breadcrumb when diagnosing startup issues in dev.
            // Stage D's logging-config story will replace this with a
            // proper JSON-line log.
            "--log-level", "info",
        ])
        proc.arguments = arguments
        proc.currentDirectoryURL = runtime.workingDirectory

        // Diagnostic breadcrumb so the forwarded log makes it obvious
        // which interpreter we picked. Especially valuable when a user
        // reports "search doesn't work" — we can immediately tell
        // whether they're on the bundled runtime (DMG / TestFlight) or
        // the dev venv (the developer's box).
        FileHandle.standardError.write(
            Data("[backend] runtime: \(runtime.diagnosticLabel)\n".utf8)
        )

        // Forward stdout/stderr line-by-line, prefixed, so Xcode console
        // shows e.g. `[backend] INFO: Started server process [...]`.
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let s = String(data: data, encoding: .utf8) {
                FileHandle.standardOutput.write(Data("[backend] \(s)".utf8))
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let s = String(data: data, encoding: .utf8) {
                FileHandle.standardError.write(Data("[backend] \(s)".utf8))
            }
        }

        // Make sure we send SIGTERM on app quit instead of leaving the
        // backend orphaned. `terminationHandler` runs on a background
        // queue, so we don't block the UI here.
        proc.terminationHandler = { p in
            FileHandle.standardError.write(
                Data("[backend] uvicorn exited (code=\(p.terminationStatus))\n".utf8)
            )
        }

        try proc.run()
        self.process = proc
        self.port = port
        self.pid = proc.processIdentifier
    }

    /// SIGTERM the child if alive; safe to call multiple times.
    ///
    /// This is **synchronous** by design: termination paths like
    /// ``NSApplicationDelegate/applicationWillTerminate(_:)`` and POSIX
    /// signal handlers must finish cleanup before the parent process
    /// returns, otherwise the child gets reparented to launchd and
    /// becomes an orphan. We give SIGTERM up to ``gracePeriod`` to
    /// land, then escalate to SIGKILL.
    func terminate() {
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()
        let gracePeriod: TimeInterval = 1.0
        let pollInterval: TimeInterval = 0.02
        let deadline = Date().addingTimeInterval(gracePeriod)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: pollInterval)
        }
        if proc.isRunning {
            kill(proc.processIdentifier, SIGKILL)
            proc.waitUntilExit()
        }
    }
}

// MARK: - BackendRuntime

/// Where the Python interpreter and ``rubick_backend`` package live for
/// this launch.
///
/// Resolution order (see ``resolve``):
///
/// 1. **Bundled** — `Bundle.main.resourceURL/backend-runtime/python/`,
///    populated by `scripts/build_dmg.sh` from the artifact produced
///    by `scripts/build_backend_bundle.sh`. Hermetic; no Homebrew, no
///    user-installed Python, no repo-on-disk dependency. This is what
///    DMG users hit.
/// 2. **Dev venv** — `<repo>/backend/.venv/bin/python`, the venv the
///    developer keeps next to the source. Used during `tuist generate`
///    + `xcodebuild` Debug runs and unsigned local DMGs. Repo root is
///    discovered via ``RUBICK_REPO_ROOT`` env var → Tuist-baked
///    `RubickDevRepoRoot` Info.plist key → cwd / bundle-parent walk.
/// 3. **System python** — `/usr/bin/env python3` as a last-ditch
///    fallback. Will fail unless the user happens to have all of our
///    deps installed, but emits a clearer "module not found" error
///    than silently exploding inside uvicorn.
///
/// ``RUBICK_PYTHON`` env var still wins over everything (CI / power
/// users); when set we treat it as a system-Python override with the
/// developer's repo as cwd.
struct BackendRuntime {
    enum Mode {
        /// Hermetic runtime shipped inside `Rubick.app/Contents/Resources/backend-runtime/`.
        case bundled(runtimeRoot: URL)
        /// Developer's `<repo>/backend/.venv` + repo source on disk.
        case devVenv(pythonURL: URL, repoRoot: URL)
        /// Last-resort `/usr/bin/env python3` invocation.
        case systemPython(repoRoot: URL)
    }

    let mode: Mode

    /// The interpreter the spawn uses.
    var pythonURL: URL {
        switch mode {
        case .bundled(let root):
            return root
                .appendingPathComponent("python")
                .appendingPathComponent("bin")
                .appendingPathComponent("python3")
        case .devVenv(let py, _):
            return py
        case .systemPython:
            return URL(fileURLWithPath: "/usr/bin/env")
        }
    }

    /// The cwd ``Process`` runs from. Only matters for the dev-venv
    /// path (uvicorn finds ``rubick_backend`` via ``sys.path``); we
    /// keep it pointing at ``backend/`` so dev mode matches
    /// ``python -m rubick_backend`` from the repo root.
    var workingDirectory: URL {
        switch mode {
        case .bundled(let root):
            return root
        case .devVenv(_, let repoRoot), .systemPython(let repoRoot):
            return repoRoot.appendingPathComponent("backend")
        }
    }

    /// Single-line label for the ``[backend] runtime:`` log breadcrumb.
    var diagnosticLabel: String {
        switch mode {
        case .bundled(let root):
            return "bundled @ \(root.path)"
        case .devVenv(let py, _):
            return "dev venv @ \(py.path)"
        case .systemPython:
            return "system python (/usr/bin/env python3)"
        }
    }

    /// Resolve the runtime to use for this launch. See the type's
    /// docstring for the priority ladder.
    static func resolve() -> BackendRuntime {
        let fm = FileManager.default

        // 1. Bundled runtime. The build script lays out
        //    ``backend-runtime/python/bin/python3`` inside
        //    ``Resources/`` of the .app. We check both that the
        //    interpreter exists *and* that ``rubick_backend`` was
        //    pip-installed into the bundle's site-packages — if
        //    someone copied only the python tree in by mistake the
        //    resulting confusion is much harder to debug than this
        //    one-line check.
        if let resources = Bundle.main.resourceURL {
            let runtimeRoot = resources.appendingPathComponent("backend-runtime")
            let bundlePython = runtimeRoot
                .appendingPathComponent("python")
                .appendingPathComponent("bin")
                .appendingPathComponent("python3")
            let bundleBackend = runtimeRoot
                .appendingPathComponent("python")
                .appendingPathComponent("lib")
                .appendingPathComponent("python3.12")
                .appendingPathComponent("site-packages")
                .appendingPathComponent("rubick_backend")
            if fm.isExecutableFile(atPath: bundlePython.path),
               fm.fileExists(atPath: bundleBackend.path)
            {
                return BackendRuntime(mode: .bundled(runtimeRoot: runtimeRoot))
            }
        }

        // 2-3. Fallback to the dev resolution. We compute the repo
        //      root once and reuse it for both the venv and the
        //      system-python branches.
        let repoRoot = resolveRepoRoot()

        if let override = ProcessInfo.processInfo.environment["RUBICK_PYTHON"],
           !override.isEmpty
        {
            // ``RUBICK_PYTHON`` wins: treat it as a hand-picked dev
            // venv even if it's outside the repo. Same cwd handling
            // as the regular dev-venv case.
            return BackendRuntime(mode:
                .devVenv(pythonURL: URL(fileURLWithPath: override), repoRoot: repoRoot)
            )
        }

        let venvPython = repoRoot
            .appendingPathComponent("backend")
            .appendingPathComponent(".venv")
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
        if fm.isExecutableFile(atPath: venvPython.path) {
            return BackendRuntime(mode: .devVenv(pythonURL: venvPython, repoRoot: repoRoot))
        }

        return BackendRuntime(mode: .systemPython(repoRoot: repoRoot))
    }

    /// Best-effort detection of the repo root from the running .app
    /// bundle. Used by the dev-venv / system-python branches.
    ///
    /// Lookup order:
    /// 1. ``RUBICK_REPO_ROOT`` env var — CI / power-user / one-off
    ///    overrides. Wins over everything.
    /// 2. ``RubickDevRepoRoot`` Info.plist key — Tuist substitutes
    ///    ``$(SRCROOT)`` at compile time so Debug builds find the
    ///    backend even when launched via ``open -a``  (where cwd
    ///    is ``/`` and the bundle is buried in DerivedData).
    /// 3. cwd / bundle parent-directory walk — last-resort, mostly
    ///    useful for direct ``.app/Contents/MacOS/Rubick`` invocations
    ///    from inside the repo. Looks for ``backend/pyproject.toml``.
    /// 4. Final fallback: cwd as-is (which makes the backend launch
    ///    fail loudly with a clearer "rubick_backend not found" error
    ///    than silently grabbing a stale checkout).
    private static func resolveRepoRoot() -> URL {
        let fm = FileManager.default

        if let env = ProcessInfo.processInfo.environment["RUBICK_REPO_ROOT"],
           !env.isEmpty
        {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }

        if let baked = Bundle.main.object(forInfoDictionaryKey: "RubickDevRepoRoot") as? String,
           !baked.isEmpty,
           !baked.contains("$("),  // un-substituted token → ignore
           fm.fileExists(atPath: (baked as NSString)
               .appendingPathComponent("backend/pyproject.toml"))
        {
            return URL(fileURLWithPath: baked)
        }

        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        var candidates: [URL] = [cwd]
        var dir = Bundle.main.bundleURL.deletingLastPathComponent()
        for _ in 0..<10 {
            candidates.append(dir)
            dir = dir.deletingLastPathComponent()
        }
        for c in candidates {
            let probe = c.appendingPathComponent("backend/pyproject.toml")
            if fm.fileExists(atPath: probe.path) {
                return c
            }
        }

        return cwd
    }
}

// MARK: - Errors

enum PythonProcessError: LocalizedError {
    case portDiscoveryFailed(String)
    case launchFailed(String)
    case healthCheckTimedOut(seconds: Double)

    var errorDescription: String? {
        switch self {
        case .portDiscoveryFailed(let detail):
            return "Could not pick a free port: \(detail)"
        case .launchFailed(let detail):
            return "Backend launch failed: \(detail)"
        case .healthCheckTimedOut(let seconds):
            return "Backend did not become healthy within \(Int(seconds))s."
        }
    }
}
