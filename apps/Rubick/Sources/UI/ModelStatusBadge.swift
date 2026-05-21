import SwiftUI

/// Small inline badge that surfaces a single ``BackendModelInfo``'s
/// download lifecycle + (where applicable) in-memory state.
///
/// Two signals encoded in one badge:
///
/// - **Dot color** = download lifecycle. Grey (absent / unknown) →
///   orange (partial, i.e. an interrupted ``snapshot_download`` left
///   a ``.incomplete`` blob behind) → blue (complete) → green
///   (complete *and* the embedder is hydrated in the running Python
///   process).
/// - **Label** = human-readable variant of the same.
///
/// The "loaded in memory" qualifier is the signal a file-system-only
/// path can't observe — only the backend knows whether
/// ``embed.is_loaded()`` is true. Models without a long-lived
/// in-process singleton report ``loadedInMemory == nil`` and the
/// badge stays blue rather than ever flipping green for them; today
/// the embedding model is the only entry, but the affordance stays
/// in case a future model card returns ``nil`` for the same reason.
///
/// Used by both ``Settings → Model`` and the Onboarding "Model setup"
/// step; centralizing the mapping here keeps the two call sites in
/// lockstep when we change the wording or add new download states.
struct ModelStatusBadge: View {
    let info: BackendModelInfo

    var body: some View {
        let style = Self.style(for: info)
        HStack(spacing: 6) {
            Circle()
                .fill(style.dotColor)
                .frame(width: 8, height: 8)
            Text(style.label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Pure (color, label) projection. ``static`` so the (extremely
    /// short) ``@ViewBuilder`` body above doesn't need a switch
    /// inside its result-builder grammar — Swift's result builder
    /// doesn't accept ``let x: Foo = ...; switch ...`` shaped code,
    /// so we extract.
    static func style(
        for info: BackendModelInfo
    ) -> (dotColor: Color, label: String) {
        switch info.downloadStatus {
        case .absent:
            return (.gray, "Not downloaded")
        case .partial:
            return (.orange, "Resuming…")
        case .complete where info.loadedInMemory == true:
            return (.green, "Loaded in memory")
        case .complete:
            return (.blue, "Downloaded")
        case .unknown:
            return (.gray, "—")
        }
    }
}
