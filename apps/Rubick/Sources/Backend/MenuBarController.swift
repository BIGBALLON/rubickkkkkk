// apps/Rubick/Sources/Backend/MenuBarController.swift
import AppKit
import Combine
import SwiftUI

/// Menu-bar status item + global ⌥+Space hotkey (single/double press).
///
/// - Single ⌥+Space: toggle Pulsar (quick-search panel)
/// - Double ⌥+Space: toggle Nebula (immersive main window)
/// - From either visible state, ⌥+Space hides everything.
@MainActor
final class MenuBarController: NSObject, NSMenuDelegate {
    private let statusItem: NSStatusItem
    private let hotkey = HotkeyService()
    private var watchSink: AnyCancellable?
    private var folderSink: AnyCancellable?

    private unowned let backend: BackendController

    /// Pulsar floating panel controller.
    private(set) var pulsarController: PulsarWindowController?

    private weak var statusMenuItem: NSMenuItem?

    init(backend: BackendController) {
        self.backend = backend
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        super.init()

        pulsarController = PulsarWindowController(backendController: backend)

        configureStatusItem()
        configureHotkey()
        observeWatchStateForBadge()
    }

    // MARK: - Status item

    private func configureStatusItem() {
        if let button = statusItem.button {
            if let img = NSImage(named: "StatusBarIcon") {
                img.isTemplate = true
                button.image = img
            } else {
                button.image = NSImage(
                    systemSymbolName: "wand.and.stars",
                    accessibilityDescription: "Rubick"
                )
                button.image?.isTemplate = true
            }
            button.toolTip = "Rubick — press ⌥+Space to search"
        }
        buildMenu()
    }

    private func buildMenu() {
        let menu = NSMenu()
        menu.delegate = self

        // Pulsar
        let pulsarItem = NSMenuItem(
            title: "Pulsar (⌥Space)",
            action: #selector(togglePulsarAction),
            keyEquivalent: ""
        )
        pulsarItem.target = self
        menu.addItem(pulsarItem)

        // Nebula
        let nebulaItem = NSMenuItem(
            title: "Nebula (⌥Space×2)",
            action: #selector(toggleNebulaAction),
            keyEquivalent: ""
        )
        nebulaItem.target = self
        menu.addItem(nebulaItem)

        menu.addItem(.separator())

        // Watch status
        let statusItem = NSMenuItem(title: watchStatusLine, action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        menu.addItem(statusItem)
        self.statusMenuItem = statusItem

        menu.addItem(.separator())

        // Settings
        let settingsItem = NSMenuItem(
            title: "Settings…",
            action: #selector(openSettingsAction),
            keyEquivalent: ","
        )
        settingsItem.target = self
        settingsItem.image = NSImage(systemSymbolName: "gearshape", accessibilityDescription: nil)
        settingsItem.image?.size = NSSize(width: 14, height: 14)
        menu.addItem(settingsItem)

        // Quit
        let quit = NSMenuItem(
            title: "Quit Rubick",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quit.target = NSApp
        menu.addItem(quit)

        self.statusItem.menu = menu
    }

    // MARK: - NSMenuDelegate

    nonisolated func menuWillOpen(_ menu: NSMenu) {
        MainActor.assumeIsolated {
            statusMenuItem?.title = watchStatusLine
        }
    }

    private var watchStatusLine: String {
        let folders = backend.watchService.folders.folders.count
        let jobs = backend.watchService.activeJobIds.count
        if folders == 0 { return "No folders watched" }
        var line = folders == 1 ? "1 folder watched" : "\(folders) folders watched"
        if jobs > 0 { line += " · indexing \(jobs)" }
        return line
    }

    private func observeWatchStateForBadge() {
        let svc = backend.watchService
        watchSink = svc.$activeJobIds
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.statusMenuItem?.title = self?.watchStatusLine ?? "" }
        folderSink = svc.folders.$folders
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.statusMenuItem?.title = self?.watchStatusLine ?? "" }
    }

    // MARK: - Mode switching

    /// Single press: if anything is visible → hide all. Otherwise → show Pulsar.
    private func handleSinglePress() {
        if pulsarController?.isVisible == true || backend.nebulaController.isVisible {
            hideAll()
        } else {
            pulsarController?.show()
        }
    }

    /// Double press: if Nebula is visible → hide all. Otherwise → show Nebula.
    private func handleDoublePress() {
        if backend.nebulaController.isVisible {
            hideAll()
        } else {
            // Hide Pulsar first if it's up
            if pulsarController?.isVisible == true {
                pulsarController?.hide()
            }
            backend.nebulaController.open()
        }
    }

    private func hideAll() {
        pulsarController?.hide()
        backend.nebulaController.close()
    }

    @objc private func togglePulsarAction() {
        pulsarController?.toggle()
    }

    @objc private func toggleNebulaAction() {
        if backend.nebulaController.isVisible {
            backend.nebulaController.close()
        } else {
            backend.nebulaController.open()
        }
    }

    @objc private func openSettingsAction() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Use menu-item simulation (showSettingsWindow: is unreliable in .accessory policy)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let knownSelectors: Set<String> = ["showSettingsWindow:", "showPreferencesWindow:"]
            let knownTitles: Set<String> = ["Settings\u{2026}", "Settings", "Preferences\u{2026}", "Preferences"]
            guard let mainMenu = NSApp.mainMenu else { return }
            for topLevel in mainMenu.items {
                guard let submenu = topLevel.submenu else { continue }
                for (index, item) in submenu.items.enumerated() {
                    if let action = item.action, knownSelectors.contains(NSStringFromSelector(action)) {
                        submenu.performActionForItem(at: index)
                        return
                    }
                }
                for (index, item) in submenu.items.enumerated() where knownTitles.contains(item.title) {
                    submenu.performActionForItem(at: index)
                    return
                }
            }
        }
    }

    // MARK: - Hotkey

    private func configureHotkey() {
        let installed = hotkey.install(
            onSinglePress: { [weak self] in
                self?.handleSinglePress()
            },
            onDoublePress: { [weak self] in
                self?.handleDoublePress()
            }
        )
        if !installed {
            FileHandle.standardError.write(
                Data("[MenuBarController] ⌥+Space already claimed by another app\n".utf8)
            )
        }
    }

    // MARK: - Cleanup

    func teardown() {
        hotkey.uninstall()
        NSStatusBar.system.removeStatusItem(statusItem)
    }
}
