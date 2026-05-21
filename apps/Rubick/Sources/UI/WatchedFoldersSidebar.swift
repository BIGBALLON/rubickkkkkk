import AppKit
import SwiftUI

/// Sidebar list of watched folders — one row per folder, no tree
/// expansion. The pre-v1.x design exposed each folder's subdirectory
/// tree as a `DisclosureGroup` so the user could scope a search down
/// to any nested subfolder; in practice that was clutter (users
/// rarely drilled past the root, and the indent + folder icons added
/// visual noise on the typical 3-folder install). v1.x simplifies to
/// "watched roots only" plus a context menu for Reveal in Finder /
/// Re-scan / Remove. Path-prefix scoping is still supported — tap
/// the row to toggle it.
///
/// Two modes via the ``pathPrefixFilter`` binding:
///
/// - **Pass a real ``@State Binding``** (``ContentView`` does this)
///   to make row taps drive a ``path_prefix`` filter on
///   ``runSearch``. The currently-selected folder is highlighted;
///   tap the same folder again to clear.
/// - **Omit the binding** (default ``.constant(nil)``, used by
///   ``Settings → Sources``) and row taps no-op silently.
///
/// Declares the service / store as ``@ObservedObject`` so SwiftUI
/// re-renders on:
/// - watch list mutations (``store.folders`` published)
/// - per-folder status changes (also via ``store.folders``)
/// - in-flight indexing jobs (``service.activeJobIds`` published)
struct WatchedFoldersSidebar: View {
    @ObservedObject var service: WatchService
    @ObservedObject var store: WatchedFoldersStore
    let enabled: Bool
    /// Binds to the parent's "active path-prefix filter" state. When
    /// set, ``runSearch`` adds ``path_prefix=<value>`` to the
    /// ``GET /search`` query. ``Settings → Sources`` uses the default
    /// ``.constant(nil)`` and row taps become silent no-ops.
    var pathPrefixFilter: Binding<String?> = .constant(nil)
    /// Show the gear-icon Settings button in the sidebar header.
    /// Main-window sidebar passes ``true`` so the user has a visible
    /// entry into Settings without remembering ⌘,; the Settings ▸
    /// Sources tab re-uses this same sidebar with ``false`` because
    /// surfacing the button there would loop back into the window the
    /// user is already standing in.
    var showSettingsButton: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(.thinMaterial)

            Divider()

            if store.folders.isEmpty {
                emptyHint
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(store.folders) { folder in
                            WatchedFolderRow(
                                folder: folder,
                                isSelected: pathPrefixFilter.wrappedValue
                                    == folder.url.standardized.path,
                                onTap: {
                                    togglePathFilter(target: folder.url)
                                },
                                onReveal: {
                                    NSWorkspace.shared.activateFileViewerSelecting(
                                        [folder.url]
                                    )
                                },
                                onRescan: {
                                    Task { await service.refreshFolder(folder) }
                                },
                                onRemove: { service.removeFolder(folder) }
                            )
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 8)
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Watched folders")
                .font(.headline)
            Spacer()
            if showSettingsButton {
                settingsButton
            }
            refreshButton
            Button {
                pickFolder()
            } label: {
                Image(systemName: "plus.circle.fill")
                    .font(.title3)
            }
            .buttonStyle(.borderless)
            .help("Add a folder for Rubick to watch & index")
            .disabled(!enabled)
        }
    }

    /// Gear icon → opens the macOS-native Settings scene with the
    /// same animation ⌘, gives you. Sits next to the refresh + add
    /// buttons because that's the cluster the user is *already*
    /// looking at when they're managing folders, so the cognitive
    /// distance from "I want to change something" to "click this"
    /// is minimal.
    @ViewBuilder
    private var settingsButton: some View {
        Button(action: Self.openSettingsScene) {
            Image(systemName: "gearshape")
                .font(.title3)
        }
        .buttonStyle(.borderless)
        .help("Open Settings (⌘,)")
    }

    /// Open the SwiftUI Settings scene by simulating a click on the
    /// matching item in the Application menu.
    ///
    /// Previous attempts (``NSApp.responds(to:)`` gate +
    /// ``sendAction(_:to: nil)`` + ``sendAction(_:to: item.target)``)
    /// all silently failed in practice. The Settings scene's
    /// receiver isn't on the standard responder chain for nil-
    /// targeted dispatch on this macOS / SwiftUI combo. The one
    /// path that **is** reliable is the same one AppKit itself
    /// uses when the user clicks the menu item:
    /// ``NSMenu.performActionForItem(at:)``. That call handles
    /// validation, target resolution, and the SwiftUI-internal
    /// dispatch as a single black box.
    ///
    /// Two passes over the menu so we cope with SwiftUI builds
    /// that either advertise the canonical selector or bind the
    /// menu item without exposing one:
    ///
    /// 1. Match on ``item.action`` selector name
    ///    (``showSettingsWindow:`` for macOS 14+,
    ///    ``showPreferencesWindow:`` for macOS 13).
    /// 2. Fall back to matching by title
    ///    (``"Settings…" / "Preferences…"``) so a future SwiftUI
    ///    rewiring still works.
    @MainActor
    private static func openSettingsScene() {
        let knownSelectorNames: Set<String> = [
            "showSettingsWindow:",
            "showPreferencesWindow:",
        ]
        let knownTitles: Set<String> = [
            "Settings…", "Settings",
            "Preferences…", "Preferences",
        ]
        guard let mainMenu = NSApp.mainMenu else { return }

        for topLevel in mainMenu.items {
            guard let submenu = topLevel.submenu else { continue }

            // Pass 1: by selector name.
            for (index, item) in submenu.items.enumerated() {
                if let action = item.action,
                   knownSelectorNames.contains(NSStringFromSelector(action))
                {
                    submenu.performActionForItem(at: index)
                    return
                }
            }
            // Pass 2: by title (fallback for SwiftUI builds that
            // don't expose a selector on the menu item).
            for (index, item) in submenu.items.enumerated()
            where knownTitles.contains(item.title)
            {
                submenu.performActionForItem(at: index)
                return
            }
        }
    }

    /// Click → ``service.refreshAll()``. The icon spins while a
    /// rescan is in flight (``service.isRefreshing``), and the
    /// button is disabled when there's nothing to rescan (backend
    /// down or no folders watched). Tooltip surfaces the effective
    /// behaviour so users on ``.manual`` know this button is the
    /// *only* path to a re-scan, while users on ``.live`` can read
    /// it as "I want fresh numbers right now".
    @ViewBuilder
    private var refreshButton: some View {
        Button {
            Task { await service.refreshAll() }
        } label: {
            Image(systemName: "arrow.clockwise")
                .font(.title3)
                .rotationEffect(.degrees(service.isRefreshing ? 360 : 0))
                .animation(
                    service.isRefreshing
                        ? .linear(duration: 1.0).repeatForever(autoreverses: false)
                        : .default,
                    value: service.isRefreshing
                )
        }
        .buttonStyle(.borderless)
        .disabled(!enabled || service.isRefreshing || store.folders.isEmpty)
        .help(
            service.isRefreshing
                ? "Re-scanning all watched folders…"
                : "Re-scan all watched folders for new or changed files"
        )
    }

    private var emptyHint: some View {
        VStack(spacing: 10) {
            Spacer(minLength: 24)
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 36))
                .foregroundStyle(.secondary)
            Text("Add a folder to begin")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Rubick will index its contents and keep them up to date as files change.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
            Button {
                pickFolder()
            } label: {
                Label("Add folder…", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!enabled)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func togglePathFilter(target: URL) {
        let p = target.standardized.path
        pathPrefixFilter.wrappedValue =
            (pathPrefixFilter.wrappedValue == p) ? nil : p
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.message = "Choose a folder for Rubick to watch."
        panel.prompt = "Add to Rubick"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            _ = service.addFolder(url)
        }
    }
}

// MARK: - Row

/// One watched-folder card. Two pieces of information dominate:
///
/// - **Top line**: folder name in callout / medium weight.
/// - **Second line**: status (idle / indexing… / N items · M chunks ·
///   3 min ago / failed: <reason>) in caption · secondary tint.
/// - **Bottom line**: the absolute path in caption2 / monospaced /
///   middle-truncated so the user always knows *where* the folder
///   actually lives without an extra click.
///
/// Tap = toggle path-prefix filter; the selected card pulses with a
/// soft accent tint so the binding state is visible at a glance.
/// Right-click → "Show in Finder / Re-scan / Remove" — the same
/// affordances the previous tree-style design carried, just promoted
/// out of the disclosure indent.
private struct WatchedFolderRow: View {
    let folder: WatchedFolder
    let isSelected: Bool
    let onTap: () -> Void
    let onReveal: () -> Void
    let onRescan: () -> Void
    let onRemove: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 10) {
                folderIcon
                VStack(alignment: .leading, spacing: 3) {
                    Text(folder.displayName)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    statusLine
                    Text(folder.displayPath)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.tint)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .background(
                isSelected
                    ? Color.accentColor.opacity(0.18)
                    : Color.secondary.opacity(0.06),
                in: RoundedRectangle(cornerRadius: 8)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(
                        isSelected
                            ? Color.accentColor.opacity(0.5)
                            : Color.clear,
                        lineWidth: 1
                    )
            )
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button {
                onReveal()
            } label: {
                Label("Show in Finder", systemImage: "folder")
            }
            Button {
                onRescan()
            } label: {
                Label("Re-scan this folder", systemImage: "arrow.clockwise")
            }
            Divider()
            Button(role: .destructive) {
                onRemove()
            } label: {
                Label("Remove from watch list", systemImage: "minus.circle")
            }
        }
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    @ViewBuilder
    private var folderIcon: some View {
        Image(systemName: "folder.fill")
            .font(.title3)
            .foregroundStyle(
                isSelected
                    ? Color.accentColor
                    : Color.accentColor.opacity(0.7)
            )
            .frame(width: 20, alignment: .center)
            .padding(.top, 1)
    }

    @ViewBuilder
    private var statusLine: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                statusIndicator
                Text(detailLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            // Live progress bar shown only when we have a real
            // ``done / total`` pair from the backend. Old backends
            // and the queued / walking phase both fall through with
            // ``nil`` and the row stays at the single-line
            // "indexing…" affordance.
            if case .indexing(_, let progress) = folder.status,
               let progress,
               progress.total > 0
            {
                ProgressView(
                    value: Double(progress.done),
                    total: Double(progress.total)
                )
                .progressViewStyle(.linear)
                .controlSize(.mini)
            }
        }
    }

    @ViewBuilder
    private var statusIndicator: some View {
        switch folder.status {
        case .indexing:
            ProgressView()
                .controlSize(.mini)
                .frame(width: 10, height: 10)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption2)
                .foregroundStyle(folder.status.tint)
                .frame(width: 10, height: 10)
        case .idle, .ready:
            Circle()
                .fill(folder.status.tint)
                .frame(width: 7, height: 7)
        }
    }

    private var detailLabel: String {
        switch folder.status {
        case .idle:
            return "idle"
        case .indexing(_, let progress):
            // Word-choice tracks whether any file in the run has
            // produced new rows yet. A pure cache-hit re-scan reads
            // "scanning… N / total" so the user knows the GPU isn't
            // working — the progress bar is just chewing through the
            // walk. Once a real embed lands the verb flips to
            // "indexing".
            if let progress, progress.total > 0 {
                let verb = progress.hasEmbedded ? "indexing" : "scanning"
                return "\(verb)… \(progress.done) / \(progress.total)"
            }
            return "scanning…"
        case .ready(let at, let stats):
            return "\(stats.files) items · \(stats.chunks) chunks · \(relative(at))"
        case .failed(let reason):
            return "failed — \(reason)"
        }
    }

    private func relative(_ d: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: d, relativeTo: Date())
    }
}
