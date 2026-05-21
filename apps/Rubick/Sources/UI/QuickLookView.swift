import AppKit
import Quartz
import SwiftUI

/// SwiftUI wrapper around AppKit's ``QLPreviewView`` — the in-window
/// equivalent of the system-wide QuickLook panel.
///
/// We deliberately use ``QLPreviewView`` (inline preview, sized to its
/// container) rather than ``QLPreviewPanel`` (shared system panel):
///
/// - **Predictable lifecycle**. ``QLPreviewPanel`` is a singleton that
///   walks the AppKit responder chain looking for a controller; mixing
///   that with SwiftUI's view lifecycle is fragile. ``QLPreviewView``
///   is a normal view we own.
/// - **Hosted in a SwiftUI sheet**. Sheet dismissal (Esc / Cmd-W /
///   click-out via our close button) is the natural exit path. No
///   double-keyboard handling, no Space-toggle race.
/// - **Same renderer under the hood**. ``QLPreviewView`` delegates to
///   the same QuickLook generator plugins as the shared panel, so
///   support for HEIC / PDF / mov / md is identical.
struct QuickLookView: NSViewRepresentable {
    let url: URL
    let title: String?

    func makeNSView(context: Context) -> QLPreviewView {
        // ``QLPreviewView.Style.normal`` matches Finder's Space-bar
        // preview; ``.compact`` is for menubar-sized hits and doesn't
        // surface format-specific controls (e.g. PDF page nav).
        let view = QLPreviewView(frame: .zero, style: .normal)!
        view.autostarts = true
        view.previewItem = PreviewItem(url: url, title: title)
        return view
    }

    func updateNSView(_ nsView: QLPreviewView, context: Context) {
        // Re-binding the item is how we get the same panel to render
        // a different file when the user navigates between hits. We
        // diff on URL because ``QLPreviewItem`` is an NSObject without
        // a built-in equality contract; URL equality is exactly what
        // QLPreviewView would have to compare under the hood anyway.
        let current = nsView.previewItem as? PreviewItem
        if current?.previewItemURL != url {
            nsView.previewItem = PreviewItem(url: url, title: title)
        }
    }

    static func dismantleNSView(_ nsView: QLPreviewView, coordinator: ()) {
        // Required to free underlying QLPreviewView resources promptly
        // when the sheet closes — without this, a renderer for e.g. a
        // long video can keep buffers around until the next GC pass.
        nsView.close()
    }
}

/// `NSObject`-backed ``QLPreviewItem`` — the only way QuickLook
/// accepts custom titles. We don't subclass ``NSURL`` (the alternative
/// "trivial" approach) because that loses the friendly title and
/// re-shows the file path in the toolbar.
private final class PreviewItem: NSObject, QLPreviewItem {
    let previewItemURL: URL?
    let previewItemTitle: String?

    init(url: URL, title: String?) {
        self.previewItemURL = url
        self.previewItemTitle = title
    }
}

// MARK: - Sheet host

/// Modal sheet that hosts a ``QuickLookView`` for one ``SearchHit``.
///
/// The sheet is small chrome — a header (filename + modality badge +
/// "Reveal in Finder" + close button) and the QuickLook viewport.
/// Sheets dismiss on Escape natively, so we don't need an explicit
/// keyboard shortcut for close.
///
/// Robustness: if the file went missing between ingest and now (user
/// deleted it from Finder), we show a graceful empty state instead
/// of the dreaded QuickLook "?" thumbnail.
struct PreviewSheet: View {
    let hit: SearchHit
    let onClose: () -> Void

    private var modality: Modality { Modality(rawString: hit.modality) }

    private var fileURL: URL? {
        guard let path = hit.filePaths.first else { return nil }
        return URL(fileURLWithPath: path)
    }

    private var fileExists: Bool {
        guard let url = fileURL else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.thinMaterial)

            Divider()

            if let url = fileURL, fileExists {
                QuickLookView(url: url, title: hit.filename)
                    .frame(minWidth: 640, idealWidth: 800, minHeight: 480, idealHeight: 560)
            } else {
                missingFile
            }
        }
        .frame(idealWidth: 800, idealHeight: 600)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: modality.symbolName)
                .foregroundStyle(modality.tint)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text(hit.filename)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let path = hit.filePaths.first {
                    Text(path)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
            if let url = fileURL, fileExists {
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                } label: {
                    Label("Show in Finder", systemImage: "folder")
                }
                .buttonStyle(.borderless)
            }
            Button(action: onClose) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .keyboardShortcut(.cancelAction)
            .help("Close (Esc)")
        }
    }

    // MARK: - Missing-file state

    private var missingFile: some View {
        VStack(spacing: 8) {
            Image(systemName: "questionmark.folder")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("File not found")
                .font(.headline)
            if let path = hit.filePaths.first {
                Text(path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
            Text("The indexed file appears to have been moved or deleted.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(minWidth: 480, minHeight: 240)
    }
}
