import AppKit
import SwiftUI

@main
struct RubickApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            SettingsView()
                .environmentObject(appDelegate.backend)
        }
    }
}

/// AppDelegate owns the BackendController and all surfaces.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    let backend = BackendController()
    private(set) var menuBar: MenuBarController?
    private var sigSources: [DispatchSourceSignal] = []
    private var onboardingWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.servicesMenu = nil
        menuBar = MenuBarController(backend: backend)
        installSignalHandlers()
        Task { await backend.start() }

        // Show onboarding on first launch
        if UserDefaults.standard.object(forKey: "onboarding_completed_at") == nil {
            showOnboarding()
        }
    }

    private func showOnboarding() {
        let onboardingView = OnboardingView {
            UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: "onboarding_completed_at")
            self.onboardingWindow?.orderOut(nil)
            self.onboardingWindow = nil
            // Open Settings → Service tab after onboarding
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                NSApp.setActivationPolicy(.regular)
                NSApp.activate(ignoringOtherApps: true)
                NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
            }
        }
        .environmentObject(backend)

        let hosting = NSHostingController(rootView: onboardingView)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Welcome to Rubick"
        window.styleMask = [.titled, .closable]
        window.setContentSize(NSSize(width: 540, height: 480))
        window.center()
        window.level = .floating
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.onboardingWindow = window
    }

    func applicationWillTerminate(_ notification: Notification) {
        menuBar?.teardown()
        backend.shutdown()
    }

    func applicationShouldHandleReopen(
        _ sender: NSApplication,
        hasVisibleWindows flag: Bool
    ) -> Bool {
        if !flag {
            backend.nebulaController.open()
        }
        return true
    }

    private func installSignalHandlers() {
        guard sigSources.isEmpty else { return }
        for sig in [SIGTERM, SIGINT] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler { [weak self] in
                self?.menuBar?.teardown()
                self?.backend.shutdown()
                signal(sig, SIG_DFL)
                raise(sig)
            }
            src.resume()
            sigSources.append(src)
        }
    }
}
