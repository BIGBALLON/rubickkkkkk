import Foundation

/// Mirrors the JSON returned by `GET /healthz`.
///
/// Kept intentionally tiny so it survives schema churn in the backend
/// without forcing a Swift recompile — the only field we actually
/// depend on is ``status == "ok"``.
struct HealthResponse: Codable, Sendable {
    let status: String
    let version: String
}

/// Mirrors the JSON returned by `GET /search` (api/search.py).
///
/// Field naming: backend uses snake_case (idiomatic Python / FastAPI);
/// Swift uses camelCase. The custom `CodingKeys` enum is the
/// single place that bridge is hard-coded.
struct SearchResponse: Codable, Sendable {
    let query: String
    let count: Int
    let tookMs: Timings
    let results: [SearchHit]

    enum CodingKeys: String, CodingKey {
        case query
        case count
        case tookMs = "took_ms"
        case results
    }

    struct Timings: Codable, Sendable {
        let embed: Double
        let search: Double
        let total: Double
    }
}

/// One row in `SearchResponse.results`.
///
/// `id` is the LanceDB primary key `<doc_id>-<modality>-<chunk_idx>`,
/// stable across queries, which lets SwiftUI's `List` / `ForEach`
/// reuse rows by identity instead of by index.
///
/// Per modality, optional fields are populated as follows:
/// - **text** rows: `rawText` (chunk preview), `thumbnailPath` nil.
/// - **image** rows: `thumbnailPath` (128-px WebP), `rawText` nil.
/// - **video** rows: `thumbnailPath` set (128-px WebP from the 1-s
///   seed frame), `rawText` nil.
///
/// Legacy `audio` / `audio_transcript` / `video_transcript` rows from
/// Legacy indexes from before v0.0.2 may still surface from old databases;
/// current ingest never writes any of them. See `Modality.swift` for
/// the deprecation note.
///
/// Hybrid retrieval fields (optional — older backends omit them):
/// - `scoreRrf` — fused RRF score (positive, scale ~0.01-0.05).
/// - `scoreVector` — cosine similarity (same value as legacy
///   `similarity`, retained for clarity).
/// - `scoreBm25` — raw BM25 score, ``nil`` if BM25 leg didn't hit.
/// - `hitCount` — number of chunks of this doc that matched. ``1``
///   for single-chunk files (markdown ≤ 600 tokens, all images, all
///   short videos). UI shows a "+N" badge when ``> 1``.
///
/// Decoding is intentionally tolerant: backends that omit hybrid fields
/// still round-trip because we fall back to defaults via `decodeIfPresent`.
struct SearchHit: Codable, Identifiable, Sendable, Hashable {
    let id: String
    let docId: String
    let modality: String
    let chunkIdx: Int
    let filePaths: [String]
    let filename: String
    let rawText: String?
    let thumbnailPath: String?
    let similarity: Double
    let scoreRrf: Double
    let scoreVector: Double?
    let scoreBm25: Double?
    let hitCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case docId = "doc_id"
        case modality
        case chunkIdx = "chunk_idx"
        case filePaths = "file_paths"
        case filename
        case rawText = "raw_text"
        case thumbnailPath = "thumbnail_path"
        case similarity
        case scoreRrf = "score_rrf"
        case scoreVector = "score_vector"
        case scoreBm25 = "score_bm25"
        case hitCount = "hit_count"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.docId = try c.decode(String.self, forKey: .docId)
        self.modality = try c.decode(String.self, forKey: .modality)
        self.chunkIdx = try c.decode(Int.self, forKey: .chunkIdx)
        self.filePaths = try c.decode([String].self, forKey: .filePaths)
        self.filename = try c.decode(String.self, forKey: .filename)
        self.rawText = try c.decodeIfPresent(String.self, forKey: .rawText)
        self.thumbnailPath = try c.decodeIfPresent(String.self, forKey: .thumbnailPath)
        self.similarity = try c.decode(Double.self, forKey: .similarity)
        // Hybrid fields default from similarity so older backends still render.
        self.scoreRrf = try c.decodeIfPresent(Double.self, forKey: .scoreRrf)
            ?? self.similarity
        self.scoreVector = try c.decodeIfPresent(Double.self, forKey: .scoreVector)
            ?? self.similarity
        self.scoreBm25 = try c.decodeIfPresent(Double.self, forKey: .scoreBm25)
        self.hitCount = try c.decodeIfPresent(Int.self, forKey: .hitCount) ?? 1
    }
}

// MARK: - Index jobs

/// Body sent to ``POST /index/job``. Each path is an absolute filesystem
/// path on the user's Mac — the backend itself runs as a local
/// subprocess so we don't need any tunneling.
struct IndexJobRequest: Codable, Sendable {
    let paths: [String]
}

/// Mirrors the JSON ``GET /index/job/{id}`` returns. We keep the
/// status as a typed enum and the stats as an optional struct so
/// SwiftUI bindings don't have to special-case "still queued".
///
/// ``progress`` is non-nil while the job is running and reports
/// ``done / total`` files plus the absolute path of the most recently
/// processed file. Old backends that don't emit this field decode to
/// nil and the UI falls back to an indeterminate spinner.
struct IndexJob: Codable, Sendable, Identifiable {
    let id: String
    let paths: [String]
    let status: Status
    /// Epoch seconds — matches the Python backend's ``int(time.time())``.
    let enqueuedAt: Int
    let startedAt: Int?
    let finishedAt: Int?
    let stats: Stats?
    let progress: Progress?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case paths
        case status
        case enqueuedAt = "enqueued_at"
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case stats
        case progress
        case error
    }

    enum Status: String, Codable, Sendable {
        case queued, running, succeeded, failed
    }

    struct Stats: Codable, Sendable, Hashable {
        let files: Int
        let chunks: Int
        let skipped: Int
    }

    /// Live ingest progress (v1.x ingest-progress wiring). Pull from
    /// ``GET /index/job/{id}`` polling. Backwards-compatible: an old
    /// backend without the field decodes to ``nil``.
    ///
    /// ``embedded`` is the subset of ``done`` that actually ran through
    /// the model (vs fast-skipped via the path+mtime cache or sha-skipped
    /// via the dedup gate). UI uses ``embedded == 0 && done > 0`` to
    /// render "Scanning…" instead of the misleading "Indexing…" copy
    /// during a pure cache-hit re-scan. Older backends decode it as ``0``.
    struct Progress: Codable, Sendable, Hashable {
        let total: Int
        let done: Int
        let current: String?
        let embedded: Int

        enum CodingKeys: String, CodingKey {
            case total, done, current, embedded
        }

        init(total: Int, done: Int, current: String?, embedded: Int = 0) {
            self.total = total
            self.done = done
            self.current = current
            self.embedded = embedded
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.total = try c.decode(Int.self, forKey: .total)
            self.done = try c.decode(Int.self, forKey: .done)
            self.current = try c.decodeIfPresent(String.self, forKey: .current)
            self.embedded = (try? c.decodeIfPresent(Int.self, forKey: .embedded)) ?? 0
        }

        /// Fraction in `[0, 1]`; `0` when ``total`` is 0 or unknown so
        /// the ``ProgressView(value:total:)`` doesn't divide by zero.
        var fraction: Double {
            total > 0 ? min(1.0, Double(done) / Double(total)) : 0
        }

        /// True when at least one file in the run produced new rows.
        /// Drives the UI's "Indexing" vs "Scanning" wording.
        var hasEmbedded: Bool { embedded > 0 }
    }
}

// MARK: - Aggregate index stats (Settings → Index tab)

/// Mirrors the JSON returned by ``GET /index/stats``.
///
/// Used by the Settings → Index tab to render "you've indexed N
/// chunks across M files, broken down by modality". Counts are
/// computed live on the LanceDB table on each request — see
/// ``store/schema.py::index_stats``. Returned modality keys are
/// alphabetized server-side so the SwiftUI ``ForEach`` renders rows
/// in stable order.
struct IndexStats: Codable, Sendable, Hashable {
    let totalChunks: Int
    let totalDocs: Int
    /// Per-modality **chunk** counts (LanceDB rows). For text — where
    /// one file produces multiple chunks — this is larger than the
    /// file count; for image / video the two counts agree.
    let byModality: [String: Int]
    /// Per-modality **item** counts (distinct ``doc_id`` values).
    /// Used for status-bar "N items · M chunks" breakdowns.
    /// Decoded lazily: older backends omitting ``by_modality_docs`` → empty dict.
    let byModalityDocs: [String: Int]

    enum CodingKeys: String, CodingKey {
        case totalChunks = "total_chunks"
        case totalDocs = "total_docs"
        case byModality = "by_modality"
        case byModalityDocs = "by_modality_docs"
    }

    init(
        totalChunks: Int,
        totalDocs: Int,
        byModality: [String: Int],
        byModalityDocs: [String: Int] = [:]
    ) {
        self.totalChunks = totalChunks
        self.totalDocs = totalDocs
        self.byModality = byModality
        self.byModalityDocs = byModalityDocs
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.totalChunks = try c.decode(Int.self, forKey: .totalChunks)
        self.totalDocs = try c.decode(Int.self, forKey: .totalDocs)
        self.byModality = try c.decode([String: Int].self, forKey: .byModality)
        // Backward compat: a v0.0.1 backend (or any future build that
        // drops the field) still round-trips because we tolerate the
        // missing key with an empty fallback. The status-bar popover
        // hides the items column when this is empty.
        self.byModalityDocs =
            (try? c.decode([String: Int].self, forKey: .byModalityDocs)) ?? [:]
    }

    /// Convenience for SwiftUI ``ForEach``: the per-modality chunks
    /// dict projected as a sorted ``[(modality, count)]`` list.
    /// Sorted by modality name to match the backend's alphabetical
    /// ordering.
    var byModalitySorted: [(modality: String, count: Int)] {
        byModality.sorted { $0.key < $1.key }.map { (modality: $0.key, count: $0.value) }
    }

    /// Per-modality breakdown as ``[(modality, items, chunks)]``.
    /// Used by the status-bar popover when ``totalDocs`` is absent.
    var byModalityWithItemsSorted: [(modality: String, items: Int, chunks: Int)] {
        let modalities = Set(byModality.keys).union(byModalityDocs.keys).sorted()
        return modalities.map { m in
            (
                modality: m,
                items: byModalityDocs[m] ?? 0,
                chunks: byModality[m] ?? 0
            )
        }
    }

    /// Empty stats — shown before the first request comes back so
    /// the UI doesn't have to special-case ``nil``.
    static let empty = IndexStats(
        totalChunks: 0,
        totalDocs: 0,
        byModality: [:],
        byModalityDocs: [:]
    )
}

// MARK: - Backend chunking settings

/// Mirrors the JSON returned by ``GET /settings`` (api/settings.py).
///
/// The backend bundles the live values together with the static
/// metadata (defaults + bounds) so the Swift Settings → Index UI can
/// drive its preset cards + clamp its custom-stepper without a
/// second round-trip. Field naming bridges snake_case → camelCase
/// the same way ``BackendModelInfo`` does.
struct BackendChunkingSettings: Codable, Sendable, Hashable {
    let targetTokens: Int
    let hardMaxTokens: Int
    /// User-defined fnmatch globs the ingest walker applies to dir +
    /// file basenames (v1.x #3). Empty means "no user rules".
    let exclusionPatterns: [String]
    let defaults: Pair
    let bounds: BoundsPair
    /// Always-on dir-name deny-list, surfaced for the Privacy UI's
    /// "Always excluded" section. Identical to the v0.0.2 hard-coded
    /// Swift mirror; sourcing it from the backend means a future
    /// addition (``vendor/`` ?) only needs a Python-side edit.
    let defaultExclusionDirNames: [String]
    /// Server-enforced caps for user pattern lists. The Swift editor
    /// mirrors them so the user gets immediate "you've hit the cap"
    /// feedback instead of finding out after a round-trip.
    let exclusionPatternLimits: PatternLimits

    struct Pair: Codable, Sendable, Hashable {
        let targetTokens: Int
        let hardMaxTokens: Int

        enum CodingKeys: String, CodingKey {
            case targetTokens = "target_tokens"
            case hardMaxTokens = "hard_max_tokens"
        }
    }

    struct PatternLimits: Codable, Sendable, Hashable {
        let maxCount: Int
        let maxLength: Int

        enum CodingKeys: String, CodingKey {
            case maxCount = "max_count"
            case maxLength = "max_length"
        }
    }

    /// Min / max bounds for each axis (the backend serialises these
    /// as 2-element arrays, e.g. ``[100, 8192]``). We expose them
    /// as a ``ClosedRange<Int>`` to make ``.clamped(to:)`` natural
    /// on the Swift side.
    struct BoundsPair: Codable, Sendable, Hashable {
        let targetTokens: ClosedRange<Int>
        let hardMaxTokens: ClosedRange<Int>

        enum CodingKeys: String, CodingKey {
            case targetTokens = "target_tokens"
            case hardMaxTokens = "hard_max_tokens"
        }

        init(targetTokens: ClosedRange<Int>, hardMaxTokens: ClosedRange<Int>) {
            self.targetTokens = targetTokens
            self.hardMaxTokens = hardMaxTokens
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.targetTokens = try Self.decodeBound(from: c, key: .targetTokens)
            self.hardMaxTokens = try Self.decodeBound(from: c, key: .hardMaxTokens)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(
                [targetTokens.lowerBound, targetTokens.upperBound],
                forKey: .targetTokens
            )
            try c.encode(
                [hardMaxTokens.lowerBound, hardMaxTokens.upperBound],
                forKey: .hardMaxTokens
            )
        }

        private static func decodeBound(
            from c: KeyedDecodingContainer<CodingKeys>,
            key: CodingKeys
        ) throws -> ClosedRange<Int> {
            let raw = try c.decode([Int].self, forKey: key)
            guard raw.count == 2, raw[0] <= raw[1] else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: c,
                    debugDescription: "expected [lo, hi] with lo ≤ hi, got \(raw)"
                )
            }
            return raw[0]...raw[1]
        }
    }

    enum CodingKeys: String, CodingKey {
        case targetTokens = "target_tokens"
        case hardMaxTokens = "hard_max_tokens"
        case exclusionPatterns = "exclusion_patterns"
        case defaults
        case bounds
        case defaultExclusionDirNames = "default_exclusion_dir_names"
        case exclusionPatternLimits = "exclusion_pattern_limits"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.targetTokens = try c.decode(Int.self, forKey: .targetTokens)
        self.hardMaxTokens = try c.decode(Int.self, forKey: .hardMaxTokens)
        // ``exclusionPatterns`` lands as part of v1.x #3 — tolerate
        // an older backend that doesn't emit the field by falling
        // back to ``[]``. Same idea for the metadata fields.
        self.exclusionPatterns =
            (try? c.decode([String].self, forKey: .exclusionPatterns)) ?? []
        self.defaults = try c.decode(Pair.self, forKey: .defaults)
        self.bounds = try c.decode(BoundsPair.self, forKey: .bounds)
        self.defaultExclusionDirNames =
            (try? c.decode([String].self, forKey: .defaultExclusionDirNames)) ?? []
        self.exclusionPatternLimits =
            (try? c.decode(PatternLimits.self, forKey: .exclusionPatternLimits))
            ?? PatternLimits(maxCount: 64, maxLength: 200)
    }

    init(
        targetTokens: Int,
        hardMaxTokens: Int,
        exclusionPatterns: [String] = [],
        defaults: Pair,
        bounds: BoundsPair,
        defaultExclusionDirNames: [String] = [],
        exclusionPatternLimits: PatternLimits =
            PatternLimits(maxCount: 64, maxLength: 200)
    ) {
        self.targetTokens = targetTokens
        self.hardMaxTokens = hardMaxTokens
        self.exclusionPatterns = exclusionPatterns
        self.defaults = defaults
        self.bounds = bounds
        self.defaultExclusionDirNames = defaultExclusionDirNames
        self.exclusionPatternLimits = exclusionPatternLimits
    }
}

/// Mirrors the JSON the Swift client sends to ``PATCH /settings``.
/// All fields optional so a UI flipping one knob doesn't have to
/// resend the others (matches backend ``ChunkingPatch``).
struct BackendChunkingPatch: Codable, Sendable, Hashable {
    let targetTokens: Int?
    let hardMaxTokens: Int?
    /// Send ``[]`` to clear all user rules; send ``nil`` to leave
    /// them unchanged (the Swift encoder skips ``nil`` fields).
    let exclusionPatterns: [String]?

    enum CodingKeys: String, CodingKey {
        case targetTokens = "target_tokens"
        case hardMaxTokens = "hard_max_tokens"
        case exclusionPatterns = "exclusion_patterns"
    }
}

/// Mirrors the JSON returned by ``PATCH /settings`` — the post-update
/// snapshot, no metadata envelope (the metadata never changes per
/// request, so we save bytes by not re-sending it).
struct BackendChunkingSnapshot: Codable, Sendable, Hashable {
    let targetTokens: Int
    let hardMaxTokens: Int
    let exclusionPatterns: [String]

    enum CodingKeys: String, CodingKey {
        case targetTokens = "target_tokens"
        case hardMaxTokens = "hard_max_tokens"
        case exclusionPatterns = "exclusion_patterns"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.targetTokens = try c.decode(Int.self, forKey: .targetTokens)
        self.hardMaxTokens = try c.decode(Int.self, forKey: .hardMaxTokens)
        self.exclusionPatterns =
            (try? c.decode([String].self, forKey: .exclusionPatterns)) ?? []
    }

    init(
        targetTokens: Int,
        hardMaxTokens: Int,
        exclusionPatterns: [String] = []
    ) {
        self.targetTokens = targetTokens
        self.hardMaxTokens = hardMaxTokens
        self.exclusionPatterns = exclusionPatterns
    }
}

/// Mirrors the JSON returned by ``DELETE /model/cache?id=<…>``
/// (v1.x #5 — Settings → Model "Re-download").
///
/// Idempotent on the backend: a delete against an already-absent
/// cache returns ``wasPresent=false`` + ``deletedBytes=0`` rather
/// than 404, so the UI's "I'll let the user re-click safely" gate
/// can be a single ``await client.clearModelCache(id:)`` without a
/// pre-flight existence check.
struct ClearModelCacheResult: Codable, Sendable, Hashable {
    /// Echoed back from the request — useful for logging and for
    /// keeping the UI's "which card am I updating?" hand-off honest.
    let id: String
    /// HuggingFace repo string (``"<owner>/<repo>"``).
    let repo: String
    /// Bytes that were under the cache subtree before delete. ``0``
    /// when ``wasPresent == false``.
    let deletedBytes: Int64
    /// Absolute path that was removed; ``nil`` when the cache was
    /// already absent (nothing to remove).
    let path: String?
    /// ``true`` if the rmtree found something to delete.
    let wasPresent: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case repo
        case deletedBytes = "deleted_bytes"
        case path
        case wasPresent = "was_present"
    }
}

/// Mirrors the JSON returned by ``GET /index/status`` /
/// ``POST /index/pause`` / ``POST /index/resume``.
///
/// Two attribute lookups on the backend (queue's pause-event +
/// ``queue.qsize() + running``), so cheap enough to refresh on every
/// Settings → Index appearance and on every job-drain edge in
/// ``BackendController``. The Swift side never polls in the
/// background — pause / resume is push-driven (button → POST →
/// refresh), and ``pending`` is kept synced via the same drain hook
/// that owns ``indexStats``.
struct IndexQueueStatus: Codable, Sendable, Hashable {
    let paused: Bool
    let pending: Int

    static let unknown = IndexQueueStatus(paused: false, pending: 0)
}

/// Mirrors the JSON returned by ``DELETE /index/by-path-prefix``.
///
/// All three counts are post-delete totals: how many chunks were
/// removed, how many distinct documents that represented, and how
/// many thumbnail files on disk were unlinked. ``deleted_thumbnails``
/// is always ≤ ``deleted_chunks`` (only image / video rows have a
/// ``thumbnail_path``; text rows contribute zero).
struct ClearByPrefixResult: Codable, Sendable, Hashable {
    let deletedChunks: Int
    let deletedDocs: Int
    let deletedThumbnails: Int?

    enum CodingKeys: String, CodingKey {
        case deletedChunks = "deleted_chunks"
        case deletedDocs = "deleted_docs"
        case deletedThumbnails = "deleted_thumbnails"
    }
}

// MARK: - Nebula M3

struct NebulaMapResponse: Codable {
    let version: Int
    let computedAt: Int
    let totalPoints: Int
    let points: [NebulaPoint]

    enum CodingKeys: String, CodingKey {
        case version
        case computedAt = "computed_at"
        case totalPoints = "total_points"
        case points
    }
}

struct NebulaPoint: Codable, Identifiable {
    let docId: String
    let chunkId: String
    let x: Float
    let y: Float
    let z: Float
    let cluster: Int?
    let modality: String
    let thumbnailPath: String?
    let filename: String

    var id: String { chunkId }

    enum CodingKeys: String, CodingKey {
        case docId = "doc_id"
        case chunkId = "chunk_id"
        case x, y, z, cluster, modality
        case thumbnailPath = "thumbnail_path"
        case filename
    }
}

struct NebulaRecomputeResponse: Codable {
    let jobId: String?
    let status: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
    }
}

struct NebulaStatusResponse: Codable {
    let state: String
    let progress: Double
    let lastComputedAt: Int
    let totalPoints: Int
    let stale: Bool

    enum CodingKeys: String, CodingKey {
        case state, progress
        case lastComputedAt = "last_computed_at"
        case totalPoints = "total_points"
        case stale
    }
}
