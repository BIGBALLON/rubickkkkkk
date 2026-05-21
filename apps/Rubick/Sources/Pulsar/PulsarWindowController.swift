// apps/Rubick/Sources/Pulsar/PulsarWindowController.swift
import AppKit
import SwiftUI

/// Manages the Pulsar floating panel lifecycle.
///
/// - `show()`: position → orderFront → focus search field
/// - `hide()`: orderOut + clear query
/// - `toggle()`: key + visible → hide, else → show
@MainActor
final class PulsarWindowController {

    private enum Layout {
        static let width: CGFloat = 528  // 36 logo overlap area + 480 panel + 12 safety
    }

    private var panel: PulsarPanel?
    private var hostingView: NSHostingView<PulsarView>?
    private let viewModel = PulsarViewModel()

    private unowned let backendController: BackendController

    init(backendController: BackendController) {
        self.backendController = backendController
    }

    var isVisible: Bool { panel?.isVisible == true }

    func toggle() {
        if let panel, panel.isVisible && panel.isKeyWindow {
            hide()
        } else {
            show()
        }
    }

    func show() {
        let panel = ensurePanel()
        viewModel.client = backendController.client
        positionPanel(panel)
        panel.orderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        panel.makeKey()
        forceFocusTextField(in: panel)
    }

    func hide() {
        panel?.orderOut(nil)
        viewModel.clearQuery()
    }

    // MARK: - Focus management

    private func forceFocusTextField(in panel: PulsarPanel) {
        attemptFocusTextField(in: panel)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self, weak panel] in
            guard let panel, panel.isVisible else { return }
            self?.attemptFocusTextField(in: panel)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self, weak panel] in
            guard let panel, panel.isVisible else { return }
            self?.attemptFocusTextField(in: panel)
        }
    }

    private func attemptFocusTextField(in panel: PulsarPanel) {
        guard let contentView = panel.contentView else { return }
        if let textField = findTextField(in: contentView) {
            panel.makeFirstResponder(textField)
        }
    }

    private func findTextField(in view: NSView) -> NSTextField? {
        if let tf = view as? NSTextField, tf.isEditable { return tf }
        for subview in view.subviews {
            if let found = findTextField(in: subview) { return found }
        }
        return nil
    }

    // MARK: - Panel setup

    @discardableResult
    private func ensurePanel() -> PulsarPanel {
        if let existing = panel { return existing }

        let pulsarView = PulsarView(viewModel: viewModel) { [weak self] in
            self?.hide()
        }

        // Use NSHostingView directly — NOT NSHostingController.
        // NSHostingController always injects an opaque system background
        // that cannot be reliably removed across macOS versions.
        // NSHostingView as a raw subview gives us full transparency control.
        let hView = NSHostingView(rootView: pulsarView)
        hView.wantsLayer = true
        hView.layer?.backgroundColor = CGColor.clear
        hView.layer?.isOpaque = false

        let newPanel = PulsarPanel(
            contentRect: NSRect(x: 0, y: 0, width: Layout.width, height: 100),
            styleMask: [],
            backing: .buffered,
            defer: false
        )
        // Set transparent content view, add hosting view as subview
        let container = NSView(frame: NSRect(x: 0, y: 0, width: Layout.width, height: 100))
        container.wantsLayer = true
        container.layer?.backgroundColor = CGColor.clear
        container.layer?.isOpaque = false
        container.autoresizesSubviews = true

        hView.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(hView)
        NSLayoutConstraint.activate([
            hView.topAnchor.constraint(equalTo: container.topAnchor),
            hView.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            hView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            hView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
        ])

        newPanel.contentView = container

        NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification,
            object: newPanel,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.viewModel.clearQuery() }
        }

        self.panel = newPanel
        self.hostingView = hView
        return newPanel
    }

    private func positionPanel(_ panel: PulsarPanel) {
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let screenFrame = screen.visibleFrame
        let x = screenFrame.midX - Layout.width / 2
        let y = screenFrame.maxY - screenFrame.height * 0.30
        panel.setFrameTopLeftPoint(NSPoint(x: x, y: y))
    }
}
