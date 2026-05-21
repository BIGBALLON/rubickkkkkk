import Foundation

/// Mirrors the JSON returned by `GET /healthz/model` (api/healthz.py).
///
/// Per-model lifecycle state for the HuggingFace-hosted models the
/// backend depends on (currently just the multimodal embedding model;
/// the response is wrapped in a ``models`` list so a future second
/// model lands without re-shaping the API). Replaces the client-side
/// HF-cache directory walk that ``Settings → Model`` used to do;
/// backend is now the single source of truth for "where is each
/// model in its lifecycle?".
///
/// Decoding is intentionally tolerant of an *unknown* download status
/// — a future backend that adds a fourth value (e.g. ``"verifying"``)
/// should not crash older clients. We map anything we don't recognize
/// to ``.unknown``; UI renders that as a neutral "—" badge.
struct HealthzModelResponse: Codable, Sendable, Hashable {
    let models: [BackendModelInfo]
}

/// One model card's worth of lifecycle state.
///
/// Field naming bridge: backend uses snake_case, Swift uses camelCase.
/// The ``CodingKeys`` enum is the single hard-coded translation point;
/// adding a new field on the Python side is a one-line change here.
struct BackendModelInfo: Codable, Sendable, Hashable, Identifiable {
    /// Stable string id (``"embedding"`` today; future models would
    /// pick their own short slug). Suitable as a SwiftUI ``ForEach``
    /// identity, hence ``Identifiable``.
    let id: String

    /// HuggingFace repo identifier (``"<owner>/<repo>"``).
    let repo: String

    /// Human-readable purpose blurb sourced from
    /// ``settings.MAIN_MODEL_PURPOSE`` (or its sibling, if a future
    /// second model gains a ``..._PURPOSE`` constant).
    let purpose: String

    /// Expected total bytes once fully downloaded. Used as the
    /// denominator when rendering "downloaded 1.7 / 2.0 GB"-style
    /// progress before ``cacheBytes`` reaches the final value.
    let declaredBytes: Int64

    /// Absolute on-disk path of the per-model HF cache dir; ``nil``
    /// when the dir doesn't exist yet (status ``.absent``).
    let cachePath: String?

    /// Bytes currently on disk for this model. Counts real blobs only
    /// (matches ``du -sh``); the snapshots/ symlinks contribute zero.
    let cacheBytes: Int64

    /// One of ``absent`` / ``partial`` / ``complete`` (or
    /// ``unknown`` for forward-compat with future backend revisions).
    let downloadStatus: BackendModelDownloadStatus

    /// ``true`` iff the backend currently holds this model in memory.
    /// ``nil`` from the backend means "no long-lived in-process
    /// singleton to introspect" — the UI renders that as a neutral
    /// "Downloaded" rather than ever flipping to "Loaded".
    let loadedInMemory: Bool?

    enum CodingKeys: String, CodingKey {
        case id
        case repo
        case purpose
        case declaredBytes = "declared_bytes"
        case cachePath = "cache_path"
        case cacheBytes = "cache_bytes"
        case downloadStatus = "download_status"
        case loadedInMemory = "loaded_in_memory"
    }
}

// MARK: - Permissions (v1.x #2)

/// Mirrors the JSON returned by ``GET /healthz/permissions``.
///
/// The top-level dict is keyed by permission name so a future
/// backend addition (Accessibility, Screen Recording, …) lands as a
/// sibling field without breaking older clients. v1 only carries
/// Full Disk Access; the Settings → General "Permissions" section
/// + the Onboarding step 5 are both gated on a single
/// ``fullDiskAccess.granted`` bit.
struct BackendPermissions: Codable, Sendable, Hashable {
    let fullDiskAccess: FullDiskAccessState

    enum CodingKeys: String, CodingKey {
        case fullDiskAccess = "full_disk_access"
    }
}

/// One TCC probe's worth of state.
///
/// ``granted`` is the boolean the UI gates on. ``probePath`` +
/// ``probeError`` are surfaced for the diagnostic copy under the
/// badge ("Tried /Library/.../TCC.db — Permission denied"), so a
/// confused user can see exactly what we attempted. ``platform``
/// is the Python ``platform.system()`` value at probe time —
/// non-``Darwin`` (e.g. a Linux dev VM) hides the entire section
/// since the answer is meaningless there.
struct FullDiskAccessState: Codable, Sendable, Hashable {
    let granted: Bool
    let probePath: String
    let probeError: String?
    let platform: String

    enum CodingKeys: String, CodingKey {
        case granted
        case probePath = "probe_path"
        case probeError = "probe_error"
        case platform
    }

    /// ``true`` when the probe ran on a Mac. The Settings /
    /// Onboarding surfaces hide themselves entirely outside this
    /// case (the "permission" concept doesn't apply on Linux).
    var isApplicable: Bool { platform == "Darwin" }
}

/// Mirrors the backend's ``Literal["absent", "partial", "complete"]``
/// type — see ``rubick_backend/model_status.py::DownloadStatus``.
///
/// ``unknown`` is a safety net for unexpected backend values: a future
/// backend that introduces e.g. ``"verifying"`` shouldn't crash older
/// clients. Add new cases here as the backend type grows.
enum BackendModelDownloadStatus: String, Codable, Sendable, Hashable, CaseIterable {
    case absent
    case partial
    case complete
    case unknown

    /// Forward-compatible decoding: anything we don't recognize maps
    /// to ``unknown`` instead of throwing. The backend is allowed to
    /// add new values without forcing a Swift recompile in lockstep.
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .unknown
    }

    /// Short label for the status badge in Settings → Model. Kept
    /// English-only for now; the rest of the Settings UI is too.
    var shortLabel: String {
        switch self {
        case .absent:   return "Not downloaded"
        case .partial:  return "Resuming…"
        case .complete: return "Downloaded"
        case .unknown:  return "—"
        }
    }
}

// MARK: - Download progress (GET /model/download-progress)

/// Mirrors the JSON returned by `GET /model/download-progress`.
struct ModelDownloadProgress: Decodable {
    let status: String          // idle | downloading | complete | error
    let downloadedBytes: Int64
    let totalBytes: Int64
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status
        case downloadedBytes = "downloaded_bytes"
        case totalBytes = "total_bytes"
        case error
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        status = try c.decode(String.self, forKey: .status)
        downloadedBytes = try c.decodeIfPresent(Int64.self, forKey: .downloadedBytes) ?? 0
        totalBytes = try c.decodeIfPresent(Int64.self, forKey: .totalBytes) ?? 0
        error = try c.decodeIfPresent(String.self, forKey: .error)
    }
}
