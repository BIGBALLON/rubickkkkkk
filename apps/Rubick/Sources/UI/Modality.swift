import SwiftUI

/// Strongly-typed view of the backend's ``modality`` string enum.
///
/// Active values from ``store/schema.py`` (``MODALITIES``):
/// ``text``, ``image``, ``video``, ``rejected``.
///
/// Legacy values still readable from old indexes (no new rows
/// produced in current builds): ``audio`` (audio modality removed),
/// Legacy ``audio_transcript`` / ``video_transcript`` (retired in v0.0.2).
/// All three fall through to ``unknown`` here so they render with a
/// neutral chip rather than crash the UI.
///
/// Each case owns its SF Symbol + accent color so views render
/// consistently — touching ``Modality`` is the single point of change
/// for visual identity.
enum Modality: String, Hashable, Sendable {
    case text
    case image
    case video
    case rejected
    case unknown

    init(rawString: String) {
        self = Modality(rawValue: rawString) ?? .unknown
    }

    /// SF Symbol name. Stable across macOS 13+ — all picks are from
    /// the base symbol set, no SF Pro 5/6 dependencies.
    var symbolName: String {
        switch self {
        case .text: return "doc.text"
        case .image: return "photo"
        case .video: return "film"
        case .rejected: return "exclamationmark.triangle"
        case .unknown: return "questionmark.square.dashed"
        }
    }

    /// Accent color used for the icon background tint and the
    /// modality badge. Picked to be distinguishable at a glance
    /// without being garish in either Light or Dark mode.
    var tint: Color {
        switch self {
        case .text: return .blue
        case .image: return .purple
        case .video: return .teal
        case .rejected: return .gray
        case .unknown: return .secondary
        }
    }

    /// Compact label for the modality badge — shown next to filename
    /// in the result row.
    var displayLabel: String {
        switch self {
        case .text: return "text"
        case .image: return "image"
        case .video: return "video"
        case .rejected: return "rejected"
        case .unknown: return "?"
        }
    }

    /// The set of backend ``modality`` strings this UI chip should
    /// match. ``video`` expands to include its legacy transcript
    /// sibling so users upgrading from v0.0.1 don't
    /// lose visibility on rows produced by the retired Whisper
    /// transcript track. New ingest never writes transcript rows, so
    /// the expansion is a no-op for fresh installs.
    var backendModalityValues: [String] {
        switch self {
        case .video: return ["video", "video_transcript"]
        case .text, .image: return [self.rawValue]
        case .rejected, .unknown:
            // These cases aren't user-facing facet chips today; the
            // sidebar exposes only ``text / image / video``. If a
            // future caller passes one of these directly, fall back
            // to the literal value so the filter still works.
            return [self.rawValue]
        }
    }
}
