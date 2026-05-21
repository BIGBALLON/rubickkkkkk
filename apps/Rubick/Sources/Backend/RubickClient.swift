import Foundation

/// Typed HTTP client for the local Python backend.
///
/// All requests are scoped to a single ``baseURL`` (loopback, port chosen
/// at backend boot). The client is intentionally stateless and cheap to
/// copy — long-lived state (process handle, status) lives in
/// ``BackendController`` instead.
///
/// We use ``URLSession.shared`` because:
/// - Requests are 127.0.0.1 only, so no proxy/cookie state to worry about
/// - SSE job progress is not wired yet (callers poll ``GET /index/job/{id}``)
struct RubickClient: Sendable {
    let baseURL: URL

    init(port: UInt16) {
        self.baseURL = URL(string: "http://127.0.0.1:\(port)")!
    }

    // MARK: - /healthz

    func healthz() async throws -> HealthResponse {
        let url = baseURL.appendingPathComponent("healthz")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/healthz")
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }

    /// Per-model lifecycle state for the embedding model.
    ///
    /// Cheap on the backend side (``stat()`` of the HF cache root +
    /// one in-process boolean lookup), so the Settings → Model tab
    /// can call this on every appearance and the Onboarding step 2
    /// can poll it at ~1 s while a download is in flight. The
    /// response is always shaped as a ``models`` list to keep
    /// future second-model additions backwards-compatible. See
    /// ``api/healthz.py::healthz_model`` for the JSON shape +
    /// ``BackendModelInfo`` for the field-by-field bridge.
    func healthzModel() async throws -> HealthzModelResponse {
        let url = baseURL.appendingPathComponent("healthz/model")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/healthz/model")
        return try JSONDecoder().decode(HealthzModelResponse.self, from: data)
    }

    /// Fetch the current macOS TCC permission probe (v1.x #2).
    ///
    /// Today only reports Full Disk Access; the Onboarding step 5
    /// and the Settings → General "Permissions" section both gate
    /// on ``permissions.fullDiskAccess.granted``. Single ``open()``
    /// on the backend side so the call is cheap enough to refresh
    /// on every panel appearance.
    func healthzPermissions() async throws -> BackendPermissions {
        let url = baseURL.appendingPathComponent("healthz/permissions")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/healthz/permissions")
        return try JSONDecoder().decode(BackendPermissions.self, from: data)
    }

    // MARK: - /search

    /// Hybrid-search the local index.
    ///
    /// - Parameters:
    ///   - q: natural-language query (will be URL-encoded automatically).
    ///   - limit: 1...50, defaults to 20 (matches the backend default).
    ///   - modality: optional single-modality filter for back-compat callers
    ///     (e.g. `"text"`). Prefer ``modalities`` for multi-select.
    ///   - modalities: optional sidebar facet selection. When non-empty the
    ///     set is encoded as a comma-separated ``modality`` query value;
    ///     the backend builds a SQL ``IN`` clause from it. The ``video``
    ///     chip expands to include the legacy ``video_transcript`` sibling
    ///     so legacy ``video_transcript`` rows aren't hidden (caller's job —
    ///     ``ContentView`` expands the video chip).
    ///   - pathPrefix: optional path-prefix filter (folder scope).
    ///   - mtimeAfter / mtimeBefore: optional file-mtime range
    ///     (POSIX epoch seconds, inclusive). Validated server-side
    ///     (negatives or inverted range → HTTP 400).
    func search(
        q: String,
        limit: Int = 20,
        modality: String? = nil,
        modalities: Set<String>? = nil,
        pathPrefix: String? = nil,
        mtimeAfter: Int? = nil,
        mtimeBefore: Int? = nil
    ) async throws -> SearchResponse {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("search"),
            resolvingAgainstBaseURL: false
        )!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "q", value: q),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        // ``modalities`` (set) wins over ``modality`` (single string)
        // when both are passed; sorting the set so the URL is stable
        // across runs (helps with debug logs and HTTP cache lookups).
        if let modalities, !modalities.isEmpty {
            let csv = modalities.sorted().joined(separator: ",")
            items.append(URLQueryItem(name: "modality", value: csv))
        } else if let modality {
            items.append(URLQueryItem(name: "modality", value: modality))
        }
        if let pathPrefix, !pathPrefix.isEmpty {
            items.append(URLQueryItem(name: "path_prefix", value: pathPrefix))
        }
        if let mtimeAfter {
            items.append(URLQueryItem(name: "mtime_after", value: String(mtimeAfter)))
        }
        if let mtimeBefore {
            items.append(URLQueryItem(name: "mtime_before", value: String(mtimeBefore)))
        }
        comps.queryItems = items
        guard let url = comps.url else {
            throw RubickClientError.invalidURL
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/search")
        return try JSONDecoder().decode(SearchResponse.self, from: data)
    }

    // MARK: - /search (POST, fused multimodal)

    /// Run a fused image-as-query against the local index.
    ///
    /// The backend's POST /search route accepts a multipart form
    /// with an optional ``q`` text part and a required ``image``
    /// attachment. The fused 768-dim vector lands in the same joint
    /// space as every indexed document, so retrieval is pure vector
    /// ANN (BM25 is deliberately skipped on the fused path —
    /// attaching an image means "find me things this looks like",
    /// not "find me files whose name matches the text").
    ///
    /// ``imageData`` is sent verbatim; the backend re-decodes via
    /// PIL so any format the Image-modality ingest pipeline accepts
    /// (PNG / JPEG / HEIC / …) also works here.
    ///
    /// Filter args mirror ``search(q:…)`` so the sidebar's
    /// path-prefix / modality / mtime facets continue to compose
    /// with image queries — useful for "show me images like this
    /// one, but only from the cat photos folder".
    func searchFused(
        q: String,
        imageData: Data,
        imageFilename: String,
        imageMimeType: String = "application/octet-stream",
        textWeight: Double = 0.5,
        limit: Int = 20,
        modality: String? = nil,
        modalities: Set<String>? = nil,
        pathPrefix: String? = nil,
        mtimeAfter: Int? = nil,
        mtimeBefore: Int? = nil
    ) async throws -> SearchResponse {
        let url = baseURL.appendingPathComponent("search")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        req.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )

        var body = Data()

        func appendField(_ name: String, _ value: String) {
            body.appendString("--\(boundary)\r\n")
            body.appendString(
                "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n"
            )
            body.appendString("\(value)\r\n")
        }

        appendField("q", q)
        appendField("text_weight", String(format: "%.3f", textWeight))
        appendField("limit", String(limit))
        if let modalities, !modalities.isEmpty {
            appendField("modality", modalities.sorted().joined(separator: ","))
        } else if let modality {
            appendField("modality", modality)
        }
        if let pathPrefix, !pathPrefix.isEmpty {
            appendField("path_prefix", pathPrefix)
        }
        if let mtimeAfter {
            appendField("mtime_after", String(mtimeAfter))
        }
        if let mtimeBefore {
            appendField("mtime_before", String(mtimeBefore))
        }

        // Image file part.
        body.appendString("--\(boundary)\r\n")
        body.appendString(
            "Content-Disposition: form-data; name=\"image\"; "
            + "filename=\"\(imageFilename)\"\r\n"
        )
        body.appendString("Content-Type: \(imageMimeType)\r\n\r\n")
        body.append(imageData)
        body.appendString("\r\n")

        // Closing boundary.
        body.appendString("--\(boundary)--\r\n")

        let (data, response) = try await URLSession.shared.upload(
            for: req,
            from: body
        )
        try Self.validateHTTP(
            response: response, data: data, endpoint: "/search (fused)"
        )
        return try JSONDecoder().decode(SearchResponse.self, from: data)
    }

    // MARK: - /index/stats

    /// Fetch aggregate index counts for the Settings → Index tab.
    ///
    /// ``pathPrefix`` (v1.x): when set, restricts counts to docs whose
    /// canonical first path starts with this string. The Watched-
    /// folders sidebar uses it to backfill each folder's
    /// ``items · chunks`` line after a fresh launch wipes the
    /// in-memory per-folder stats.
    ///
    /// Cheap on small tables (``count_rows`` per modality + one
    /// ``to_pandas`` for distinct-doc count); the backend wraps the
    /// call in ``asyncio.to_thread`` so it doesn't block in-flight
    /// queries.
    func indexStats(pathPrefix: String? = nil) async throws -> IndexStats {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("index/stats"),
            resolvingAgainstBaseURL: false
        )!
        if let pathPrefix, !pathPrefix.isEmpty {
            comps.queryItems = [URLQueryItem(name: "path_prefix", value: pathPrefix)]
        }
        guard let url = comps.url else {
            throw RubickClientError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/stats")
        return try JSONDecoder().decode(IndexStats.self, from: data)
    }

    /// Bulk-delete every chunk whose canonical first path starts with
    /// ``prefix``. Used when re-indexing a single watched folder (chunk purge by path);
    /// future "Re-index this folder" affordances will also call it.
    ///
    /// The backend rejects empty / single-char / ``/`` prefixes with
    /// HTTP 400 — surfaced as ``RubickClientError.httpStatus`` with the
    /// helper's message verbatim.
    func clearByPathPrefix(_ prefix: String) async throws -> ClearByPrefixResult {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("index/by-path-prefix"),
            resolvingAgainstBaseURL: false
        )!
        comps.queryItems = [URLQueryItem(name: "prefix", value: prefix)]
        guard let url = comps.url else {
            throw RubickClientError.invalidURL
        }
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/by-path-prefix")
        return try JSONDecoder().decode(ClearByPrefixResult.self, from: data)
    }

    /// Drop the entire index — all chunks, all modalities, thumbnails, nebula map.
    func clearAllIndex() async throws -> ClearByPrefixResult {
        let url = baseURL.appendingPathComponent("index/all")
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/all")
        return try JSONDecoder().decode(ClearByPrefixResult.self, from: data)
    }

    // MARK: - /model/cache (v1.x #5 — Re-download)

    /// Wipe the on-disk HuggingFace cache subtree for a model.
    ///
    /// Returns the post-delete snapshot so the UI can render
    /// "freed 1.8 GB" without a second round-trip. Idempotent —
    /// calling against an already-absent cache returns
    /// ``wasPresent=false`` + ``deletedBytes=0`` (HTTP 200), so a
    /// double-click on the Settings button doesn't surface a 404.
    ///
    /// The backend leaves the in-process model singleton alone; the
    /// user re-launches Rubick to trigger a fresh
    /// ``snapshot_download``. See ``model_status.delete_model_cache``
    /// for the rationale (no inter-thread MLX dance, no stop-the-
    /// world during inflight searches).
    ///
    /// - Throws: ``RubickClientError.httpStatus`` for 404 (unknown
    ///   model id) or 5xx (permission / IO failure under the cache
    ///   root).
    func clearModelCache(id: String) async throws -> ClearModelCacheResult {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("model/cache"),
            resolvingAgainstBaseURL: false
        )!
        comps.queryItems = [URLQueryItem(name: "id", value: id)]
        guard let url = comps.url else {
            throw RubickClientError.invalidURL
        }
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/model/cache")
        return try JSONDecoder().decode(ClearModelCacheResult.self, from: data)
    }

    // MARK: - /model/download + progress

    func triggerModelDownload(endpoint: String) async throws -> [String: String] {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("model/download"),
            resolvingAgainstBaseURL: false
        )!
        comps.queryItems = [URLQueryItem(name: "endpoint", value: endpoint)]
        guard let url = comps.url else {
            throw RubickClientError.invalidURL
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/model/download")
        let dict = try JSONDecoder().decode([String: String].self, from: data)
        return dict
    }

    func modelDownloadProgress() async throws -> ModelDownloadProgress {
        let url = baseURL.appendingPathComponent("model/download-progress")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/model/download-progress")
        return try JSONDecoder().decode(ModelDownloadProgress.self, from: data)
    }

    // MARK: - /settings

    /// Fetch the live chunking settings + UI metadata. Cheap (the
    /// backend just snapshots two ints + appends static defaults +
    /// bounds), so the Settings → Index "Text chunking" section can
    /// call this on every appearance without worrying about cost.
    func getSettings() async throws -> BackendChunkingSettings {
        let url = baseURL.appendingPathComponent("settings")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/settings")
        return try JSONDecoder().decode(BackendChunkingSettings.self, from: data)
    }

    /// Mutate one or more chunking / exclusion knobs. Returns the
    /// post-update snapshot — the backend silently clamps out-of-
    /// range tokens, silently drops invalid / duplicate exclusion
    /// entries, and trims the list to the advertised cap, so the
    /// UI must reflect what was *actually* applied (not what was
    /// requested).
    ///
    /// All arguments are optional so a single-knob flip doesn't have
    /// to resend the unchanged values. ``exclusionPatterns = []``
    /// is the explicit "clear all user rules" signal; ``nil`` means
    /// "leave them alone".
    func updateSettings(
        targetTokens: Int? = nil,
        hardMaxTokens: Int? = nil,
        exclusionPatterns: [String]? = nil
    ) async throws -> BackendChunkingSnapshot {
        let url = baseURL.appendingPathComponent("settings")
        var req = URLRequest(url: url)
        req.httpMethod = "PATCH"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(
            BackendChunkingPatch(
                targetTokens: targetTokens,
                hardMaxTokens: hardMaxTokens,
                exclusionPatterns: exclusionPatterns
            )
        )
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/settings")
        return try JSONDecoder().decode(BackendChunkingSnapshot.self, from: data)
    }

    // MARK: - /index/{pause,resume,status} (v1.x: ingest pause toggle)

    /// Stop the backend worker from picking up new ingest jobs.
    ///
    /// In-flight ingest finishes; only the next ``queue.get()`` blocks
    /// until ``resumeIndex()`` flips it back. Jobs enqueued while
    /// paused (FSEvents callbacks, manual "Re-scan" clicks) accumulate
    /// in the queue and drain in submission order once resumed.
    ///
    /// Returns the live ``{paused, pending}`` snapshot so the UI can
    /// render the "N waiting" hint next to the Resume button without
    /// a second round-trip.
    func pauseIndex() async throws -> IndexQueueStatus {
        try await postIndexLifecycle(path: "index/pause")
    }

    /// Allow the worker to fetch the next queued job. Idempotent.
    func resumeIndex() async throws -> IndexQueueStatus {
        try await postIndexLifecycle(path: "index/resume")
    }

    /// Cheap polling endpoint for "is indexing paused?". Two
    /// attribute lookups on the backend — separate from ``indexStats``
    /// because pause / resume can flip independently of row counts.
    func indexQueueStatus() async throws -> IndexQueueStatus {
        let url = baseURL.appendingPathComponent("index/status")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/status")
        return try JSONDecoder().decode(IndexQueueStatus.self, from: data)
    }

    private func postIndexLifecycle(path: String) async throws -> IndexQueueStatus {
        let url = baseURL.appendingPathComponent(path)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/\(path)")
        return try JSONDecoder().decode(IndexQueueStatus.self, from: data)
    }

    // MARK: - /index/job

    /// Enqueue an asynchronous ingest job on the backend.
    ///
    /// The returned ``IndexJob`` is in ``queued`` state; the caller
    /// must poll ``indexJobStatus(id:)`` (or use ``waitForIndexJob``)
    /// to observe progress.
    ///
    /// - Throws: ``RubickClientError.httpStatus`` for 4xx/5xx
    ///   responses (most commonly 422 if a path doesn't exist on
    ///   disk *now*; the backend stat-checks every path on enqueue).
    func enqueueIndexJob(paths: [URL]) async throws -> IndexJob {
        let url = baseURL.appendingPathComponent("index/job")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // ``URL.path`` is already percent-decoded — exactly what
        // the backend wants for ``os.path.expanduser`` / ``exists``.
        let body = IndexJobRequest(paths: paths.map(\.path))
        req.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/job")
        return try JSONDecoder().decode(IndexJob.self, from: data)
    }

    /// One-shot status fetch. Returns ``nil`` on 404 (job evicted
    /// from the backend's 256-entry history); throws on other errors.
    func indexJobStatus(id: String) async throws -> IndexJob? {
        let url = baseURL.appendingPathComponent("index/job/\(id)")
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, http.statusCode == 404 {
            return nil
        }
        try Self.validateHTTP(response: response, data: data, endpoint: "/index/job/\(id)")
        return try JSONDecoder().decode(IndexJob.self, from: data)
    }

    /// Poll a job to terminal status (``succeeded`` / ``failed``) or
    /// timeout. ``onProgress`` is invoked once per poll so the UI can
    /// surface intermediate "still running…" feedback.
    ///
    /// - Throws: ``RubickClientError.indexJobTimedOut`` on timeout,
    ///   ``RubickClientError.indexJobLost`` on 404, propagates
    ///   transport errors verbatim.
    func waitForIndexJob(
        id: String,
        pollInterval: TimeInterval = 0.5,
        timeout: TimeInterval = 3600,
        onProgress: (@Sendable (IndexJob) -> Void)? = nil
    ) async throws -> IndexJob {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            guard let job = try await indexJobStatus(id: id) else {
                throw RubickClientError.indexJobLost(id: id)
            }
            onProgress?(job)
            if job.status == .succeeded || job.status == .failed {
                return job
            }
            try await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
        }
        throw RubickClientError.indexJobTimedOut(id: id, seconds: timeout)
    }

    // MARK: - /nebula (M3)

    /// Fetch the precomputed UMAP 3-D map of image/video stars.
    func nebulaMap() async throws -> NebulaMapResponse {
        let url = baseURL.appendingPathComponent("nebula/map")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/nebula/map")
        return try JSONDecoder().decode(NebulaMapResponse.self, from: data)
    }

    /// Trigger an async UMAP recompute. Returns immediately.
    func nebulaRecompute() async throws -> NebulaRecomputeResponse {
        let url = baseURL.appendingPathComponent("nebula/recompute")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: req)
        try Self.validateHTTP(response: response, data: data, endpoint: "/nebula/recompute")
        return try JSONDecoder().decode(NebulaRecomputeResponse.self, from: data)
    }

    /// Poll compute state and staleness.
    func nebulaStatus() async throws -> NebulaStatusResponse {
        let url = baseURL.appendingPathComponent("nebula/status")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.validateHTTP(response: response, data: data, endpoint: "/nebula/status")
        return try JSONDecoder().decode(NebulaStatusResponse.self, from: data)
    }

    // MARK: - Internal helpers

    /// Surface 4xx/5xx as typed errors with the backend's error body.
    ///
    /// FastAPI returns `{"detail": "..."}` for `HTTPException`; we
    /// thread that detail string through to the UI so the user actually
    /// sees what went wrong instead of "Operation failed".
    private static func validateHTTP(
        response: URLResponse,
        data: Data,
        endpoint: String
    ) throws {
        guard let http = response as? HTTPURLResponse else {
            throw RubickClientError.transport("non-HTTP response for \(endpoint)")
        }
        guard (200...299).contains(http.statusCode) else {
            let detail = Self.extractFastAPIDetail(from: data)
                ?? String(data: data, encoding: .utf8)
                ?? "<no body>"
            throw RubickClientError.httpStatus(
                code: http.statusCode,
                endpoint: endpoint,
                detail: detail
            )
        }
    }

    private static func extractFastAPIDetail(from data: Data) -> String? {
        struct DetailEnvelope: Decodable { let detail: String }
        return (try? JSONDecoder().decode(DetailEnvelope.self, from: data))?.detail
    }
}

// MARK: - Data helpers

private extension Data {
    /// Append a UTF-8-encoded string. Used by the multipart upload
    /// path to keep ``body.append(...)`` calls terse and
    /// type-uniform — chunks of header text alternating with binary
    /// payload chunks.
    mutating func appendString(_ s: String) {
        if let data = s.data(using: .utf8) {
            self.append(data)
        }
    }
}

// MARK: - Errors

enum RubickClientError: LocalizedError {
    case invalidURL
    case transport(String)
    case httpStatus(code: Int, endpoint: String, detail: String)
    case indexJobLost(id: String)
    case indexJobTimedOut(id: String, seconds: TimeInterval)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Could not build the request URL."
        case .transport(let detail):
            return detail
        case .httpStatus(let code, let endpoint, let detail):
            return "Backend \(endpoint) → HTTP \(code): \(detail)"
        case .indexJobLost(let id):
            return "Index job \(id) disappeared from the backend "
                + "(history evicted or server restarted)."
        case .indexJobTimedOut(let id, let seconds):
            return "Index job \(id) did not finish within \(Int(seconds))s."
        }
    }
}
