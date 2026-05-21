import AppKit

/// Pulsar's floating NSPanel — doesn't steal App activation, auto-
/// dismisses when the user clicks outside.
///
/// - `styleMask`: nonactivatingPanel + fullSizeContentView + borderless
/// - `level`: modalPanel — floats above all normal windows
/// - `collectionBehavior`: canJoinAllSpaces + fullScreenAuxiliary —
///    visible even over a full-screen app
/// - `resignKey()` override → `orderOut(nil)` — clicking outside
///    immediately dismisses the panel
@MainActor
final class PulsarPanel: NSPanel {

    override init(
        contentRect: NSRect,
        styleMask style: NSWindow.StyleMask,
        backing backingStoreType: NSWindow.BackingStoreType,
        defer flag: Bool
    ) {
        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .fullSizeContentView, .borderless],
            backing: .buffered,
            defer: false
        )
        configure()
    }

    private func configure() {
        level = .modalPanel
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isMovableByWindowBackground = true
        backgroundColor = .clear
        isOpaque = false
        // Use native window shadow — macOS renders it based on opaque pixels,
        // so it hugs the rounded panel shape perfectly without any background bleed.
        hasShadow = true
        // Prevent the Dock icon from bouncing
        hidesOnDeactivate = false
        // Allow key events (needed for keyboard navigation)
        becomesKeyOnlyIfNeeded = false
    }

    /// Auto-hide when the panel loses key status (click outside, Esc).
    override func resignKey() {
        super.resignKey()
        orderOut(nil)
    }

    /// NSPanel default allows becoming key window.
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}
