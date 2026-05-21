import AppKit
import SwiftUI

/// Pulsar main view — full content of the floating panel.
///
/// Panel structure:
/// ```
/// PanelShell (gradient border + glass bg + sheen)
///   ├── PulsarSearchBar
///   ├── Result area
///   │   ├── .empty    → nothing
///   │   ├── .loading  → PulsarSkeletonView
///   │   ├── .results  → 5 × PulsarResultRow (virtual viewport)
///   │   ├── .noResults → no-results hint
///   │   └── .error    → error hint
///   └── PulsarHintBar (shown when results are present)
/// ```
///
/// Keyboard navigation via `background(KeyEventHandler{…})` + NSEvent
/// local monitor, keeping the SwiftUI view tree pure value-type.
struct PulsarView: View {

    @ObservedObject var viewModel: PulsarViewModel

    /// Esc dismisses Pulsar.
    var onDismiss: () -> Void

    @FocusState private var searchFocused: Bool
    @State private var isVisible = false

    var body: some View {
        HStack(alignment: .bottom, spacing: -10) {
            // Rubick mascot — bottom-aligned, extends above panel
            pulsarLogo

            // Main panel
            panelShell
                .frame(width: 480)
                .fixedSize(horizontal: false, vertical: true)
        }
        .scaleEffect(isVisible ? 1 : 0.95)
        .opacity(isVisible ? 1 : 0)
        .onAppear {
            withAnimation(.spring(response: 0.18, dampingFraction: 0.85)) {
                isVisible = true
            }
            searchFocused = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                searchFocused = true
            }
        }
        .onDisappear {
            isVisible = false
        }
        .background(
            KeyEventHandler(
                onKeyDown: handleKeyDown(_:)
            )
        )
    }

    // MARK: - Logo

    private var pulsarLogo: some View {
        Group {
            if let logoURL = Bundle.main.url(forResource: "rubick-logo", withExtension: "png"),
               let nsImage = NSImage(contentsOf: logoURL) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFit()
            } else {
                Image(systemName: "sparkle")
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(DS.accent)
            }
        }
        .frame(width: 64, height: 64)
    }

    // MARK: - Panel shell (glass panel — NO opaque corners)

    private var panelShell: some View {
        panelContent
            // Draw background AS a rounded rect shape — never leaks at corners
            .background(
                RoundedRectangle(cornerRadius: DS.cornerRadius, style: .continuous)
                    .fill(PulsarColors.panelFill)
            )
            .clipShape(RoundedRectangle(cornerRadius: DS.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: DS.cornerRadius, style: .continuous)
                    .strokeBorder(PulsarColors.panelBorder, lineWidth: DS.borderWidth)
            )
            // No SwiftUI .shadow() — we use the native window shadow (hasShadow)
            // which renders correctly without background bleed on transparent panels.
    }

    // MARK: - Panel content

    private var panelContent: some View {
        VStack(spacing: 0) {
            PulsarSearchBar(query: $viewModel.query, isFocused: $searchFocused)

            // Image paste chip (shown when an image is pasted)
            if let img = viewModel.pastedImage {
                imageChip(img)
            }

            resultArea

            if case .results = viewModel.phase {
                PulsarHintBar(
                    resultCount: viewModel.results.count,
                    totalCount: viewModel.totalCount
                )
            }
        }
    }

    /// Compact chip showing the pasted image with a dismiss button.
    private func imageChip(_ attachment: PulsarImageAttachment) -> some View {
        HStack(spacing: 8) {
            // Tiny thumbnail preview
            if let nsImage = NSImage(data: attachment.data) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 24, height: 24)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
            }
            Text("Image · \(attachment.formattedSize)")
                .font(.system(size: 11))
                .foregroundStyle(Color.white.opacity(0.50))
            Spacer()
            // Remove button
            Button {
                viewModel.pastedImage = nil
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(Color.white.opacity(0.30))
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.03))
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.white.opacity(0.05))
                .frame(height: 0.5)
        }
    }

    // MARK: - Result area

    @ViewBuilder
    private var resultArea: some View {
        switch viewModel.phase {
        case .empty:
            EmptyView()

        case .loading:
            PulsarSkeletonView()
                .transition(.opacity)

        case .results:
            resultsList
                .transition(.opacity)

        case .noResults:
            noResultsView
                .transition(.opacity)

        case .error(let msg):
            errorView(message: msg)
                .transition(.opacity)
        }
    }

    private var resultsList: some View {
        VStack(spacing: 0) {
            ForEach(Array(viewModel.visibleResults.enumerated()), id: \.element.id) { idx, hit in
                PulsarResultRow(
                    hit: hit,
                    index: viewModel.viewportStart + idx,
                    isSelected: idx == viewModel.selectedIndexInViewport
                )
                .onTapGesture {
                    viewModel.selectIndex(viewModel.viewportStart + idx)
                    openSelected()
                }
                // Row separator
                if idx < viewModel.visibleResults.count - 1 {
                    Rectangle()
                        .fill(Color.white.opacity(0.04))
                        .frame(height: 0.5)
                        .padding(.horizontal, 18)
                }
            }
        }
        .padding(.top, 7)
    }

    private var noResultsView: some View {
        VStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 24, weight: .thin))
                .foregroundStyle(Color.white.opacity(0.15))
            Text("No results")
                .font(.system(size: 13))
                .foregroundStyle(Color.white.opacity(0.25))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
    }

    private func errorView(message: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 20, weight: .thin))
                .foregroundStyle(Color.orange.opacity(0.5))
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(Color.white.opacity(0.30))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 28)
        .padding(.horizontal, 24)
    }

    // MARK: - Keyboard handling

    private func handleKeyDown(_ event: NSEvent) -> Bool {
        let cmd = event.modifierFlags.contains(.command)

        switch event.keyCode {
        case 125: // Down arrow
            viewModel.moveSelection(by: 1)
            return true
        case 126: // Up arrow
            viewModel.moveSelection(by: -1)
            return true
        case 36: // Return — submit search or open selected result
            if viewModel.queryChanged || viewModel.results.isEmpty {
                viewModel.submitSearch()
            } else {
                openSelected()
            }
            return true
        case 53: // Esc
            onDismiss()
            return true
        case 9: // V key — ⌘V → paste image from clipboard
            if cmd {
                if viewModel.pasteFromClipboard() {
                    return true
                }
            }
            return false
        default:
            // ⌘1 … ⌘5 — select visible row by index
            if cmd,
               let char = event.characters,
               let digit = Int(char),
               (1...5).contains(digit) {
                let idx = digit - 1
                if idx < viewModel.visibleResults.count {
                    viewModel.selectAndOpen(at: idx)
                    openSelected()
                }
                return true
            }
            return false
        }
    }

    private func openSelected() {
        guard let hit = viewModel.selectedHit,
              let path = hit.filePaths.first else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
        onDismiss()
    }
}

// MARK: - NSEvent keyboard bridge

/// Wraps an NSEvent local monitor as a SwiftUI view for key capture.
private struct KeyEventHandler: NSViewRepresentable {

    var onKeyDown: (NSEvent) -> Bool

    func makeNSView(context: Context) -> KeyCapturingView {
        let view = KeyCapturingView()
        view.onKeyDown = onKeyDown
        // Ensure zero-size, fully transparent — must not draw anything.
        view.frame = .zero
        view.wantsLayer = true
        view.layer?.backgroundColor = .clear
        return view
    }

    func updateNSView(_ nsView: KeyCapturingView, context: Context) {
        nsView.onKeyDown = onKeyDown
    }
}

final class KeyCapturingView: NSView {

    var onKeyDown: ((NSEvent) -> Bool)?
    private var monitor: Any?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // Reset monitor on every window attach/detach for state consistency.
        if let monitor {
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }
        if window != nil {
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self,
                      let win = self.window,
                      win.isVisible,
                      win.isKeyWindow else {
                    return event
                }
                if self.onKeyDown?(event) == true {
                    return nil // Event consumed
                }
                return event
            }
        }
    }

    override var acceptsFirstResponder: Bool { false }
}
