// apps/Rubick/Sources/Nebula/NebulaWindowController.swift
import AppKit
import SwiftUI

/// Manages the Nebula window — 70% screen, centered. Main application window.
@MainActor
final class NebulaWindowController {

    private var windowController: NSWindowController?
    private var viewModel: NebulaViewModel?
    private var searchViewModel: NebulaSearchViewModel?

    private unowned let backendController: BackendController

    init(backendController: BackendController) {
        self.backendController = backendController
    }

    var isVisible: Bool { windowController?.window?.isVisible == true }

    func open() {
        if let wc = windowController, let win = wc.window, win.isVisible {
            win.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        guard let client = backendController.client else { return }

        let vm = NebulaViewModel(client: client)
        self.viewModel = vm

        let searchVM = NebulaSearchViewModel()
        searchVM.client = client
        searchVM.nebulaViewModel = vm
        self.searchViewModel = searchVM

        let screen = NSScreen.main ?? NSScreen.screens.first!
        let screenFrame = screen.visibleFrame
        let scale: CGFloat = 0.7
        let w = screenFrame.width * scale
        let h = screenFrame.height * scale
        let x = screenFrame.origin.x + (screenFrame.width - w) / 2
        let y = screenFrame.origin.y + (screenFrame.height - h) / 2
        let windowFrame = NSRect(x: x, y: y, width: w, height: h)

        let watchedFolders = backendController.watchService.folders.folders.map(\.url)

        let nebulaView = NebulaView(
            viewModel: vm,
            searchViewModel: searchVM,
            watchedFolders: watchedFolders,
            onDismiss: { [weak self] in
                self?.close()
            }
        )
        .frame(width: w, height: h)

        let hosting = NSHostingController(rootView: nebulaView)
        hosting.sizingOptions = []
        hosting.view.frame = NSRect(x: 0, y: 0, width: w, height: h)

        let window = NSWindow(
            contentRect: windowFrame,
            styleMask: [.titled, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = hosting
        window.title = "Nebula"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        // Hide traffic light buttons for immersive experience
        window.standardWindowButton(.closeButton)?.isHidden = true
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true
        window.backgroundColor = NSColor(red: 0.008, green: 0.016, blue: 0.032, alpha: 1.0)
        window.isOpaque = true
        window.hasShadow = true
        window.level = .normal
        window.setFrame(windowFrame, display: true)

        let wc = NSWindowController(window: window)
        self.windowController = wc

        wc.showWindow(nil)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        // loadMap is handled inside NebulaView.makeNSView
        Task { await vm.checkStatus() }
    }

    func close() {
        windowController?.window?.orderOut(nil)
        windowController = nil
        viewModel = nil
        searchViewModel = nil
    }
}
