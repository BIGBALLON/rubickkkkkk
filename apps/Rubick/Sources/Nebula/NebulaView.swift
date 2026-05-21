// apps/Rubick/Sources/Nebula/NebulaView.swift
import Combine
import SwiftUI
import WebKit

/// Nebula — the immersive main experience.
///
/// Single layer: WKWebView (Three.js 3D star map + search UI in HTML).
/// All search UI is handled inside nebula.html via JS ↔ Swift message passing.
struct NebulaView: View {
    @ObservedObject var viewModel: NebulaViewModel
    @ObservedObject var searchViewModel: NebulaSearchViewModel
    let watchedFolders: [URL]
    let onDismiss: () -> Void

    var body: some View {
        NebulaWebView(
            viewModel: viewModel,
            searchViewModel: searchViewModel,
            onDismiss: onDismiss
        )
        .ignoresSafeArea()
        .background(Color.clear)
    }
}

// MARK: - WKWebView wrapper (with search message handler)

struct NebulaWebView: NSViewRepresentable {
    @ObservedObject var viewModel: NebulaViewModel
    @ObservedObject var searchViewModel: NebulaSearchViewModel
    let onDismiss: () -> Void

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        config.userContentController.add(context.coordinator, name: "rubickNebula")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        webView.navigationDelegate = context.coordinator
        context.coordinator.webView = webView
        context.coordinator.viewModel = viewModel
        context.coordinator.searchViewModel = searchViewModel
        context.coordinator.onDismiss = onDismiss

        // Load data: either immediately if available, or after fetching
        let coordinator = context.coordinator
        let vm = viewModel
        Task { @MainActor in
            // If stars empty, fetch from backend first
            if vm.stars.isEmpty {
                await vm.loadMap()
            }
            // Now load into WebView (with whatever data we have)
            coordinator.loadWithData(vm.stars, in: webView)
        }

        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        // Reload when star count changes (e.g. after background recompute)
        let count = viewModel.stars.count
        if count > 0 && count != context.coordinator.loadedStarCount {
            context.coordinator.loadWithData(viewModel.stars, in: webView)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        weak var webView: WKWebView?
        weak var viewModel: NebulaViewModel?
        weak var searchViewModel: NebulaSearchViewModel?
        var onDismiss: (() -> Void)?
        var hasLoaded = false
        var loadedStarCount = 0
        private var resultObservation: AnyCancellable?

        // MARK: - WKScriptMessageHandler (unified message handler)

        nonisolated func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            // WebKit guarantees this callback fires on the main thread,
            // so we can safely cross the isolation boundary here without
            // hopping back via DispatchQueue.main.async (which would
            // also fail @Sendable for the `Any` payload).
            MainActor.assumeIsolated {
                guard message.name == "rubickNebula",
                      let body = message.body as? [String: Any],
                      let type = body["type"] as? String else { return }
                handleMessage(type: type, body: body)
            }
        }

        @MainActor private func handleMessage(type: String, body: [String: Any]) {
            switch type {
            case "search":
                let query = body["query"] as? String ?? ""
                let modalities = body["modalities"] as? [String] ?? []
                performSearch(query: query, modalities: modalities)

            case "clearSearch":
                searchViewModel?.clearSearch()
                pushResultsToJS(results: [], phase: "idle")
                pushHighlights(ids: [])

            case "pasteImage":
                handlePasteImage(body: body)

            case "removeImage":
                searchViewModel?.imageAttachment = nil

            case "openFile":
                if let path = body["path"] as? String {
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                }

            case "openSettings":
                NSLog("[Nebula] Settings button pressed — opening settings")
                NSApp.setActivationPolicy(.regular)
                NSApp.activate(ignoringOtherApps: true)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    Self.openSettingsViaMenu()
                }

            case "dismiss":
                onDismiss?()

            default:
                NSLog("[Nebula] Unknown message type: %@", type)
            }
        }

        // MARK: - Search execution

        @MainActor private func performSearch(query: String, modalities: [String]) {
            guard let searchVM = searchViewModel else { return }

            // Update filters
            searchVM.modalityFilter = Set(modalities)

            // Set query and trigger search explicitly
            searchVM.query = query
            searchVM.submitSearch()

            // Observe results via Combine
            resultObservation?.cancel()
            resultObservation = searchVM.$phase
                .dropFirst() // skip current value
                .receive(on: DispatchQueue.main)
                .sink { [weak self] (phase: NebulaSearchViewModel.Phase) in
                    guard let self, let vm = self.searchViewModel else { return }
                    switch phase {
                    case .idle:
                        self.pushResultsToJS(results: [], phase: "idle")
                        self.pushHighlights(ids: [])
                    case .searching:
                        break // JS already shows spinner
                    case .results:
                        let jsonResults = vm.results.map { hit in
                            self.hitToDict(hit)
                        }
                        self.pushResultsToJS(results: jsonResults, phase: "results")
                        self.pushHighlights(ids: Array(vm.matchedStarIds))
                    case .noResults:
                        self.pushResultsToJS(results: [], phase: "noResults")
                        self.pushHighlights(ids: [])
                    case .error(let msg):
                        self.pushErrorToJS(msg)
                        self.pushHighlights(ids: [])
                    }
                }
        }

        // MARK: - Image paste for I+T fused search

        @MainActor private func handlePasteImage(body: [String: Any]) {
            guard let base64 = body["data"] as? String,
                  let mimeType = body["mimeType"] as? String,
                  let data = Data(base64Encoded: base64) else { return }

            // Convert to PNG if needed (backend expects standard formats)
            let finalData: Data
            let finalMime: String
            if mimeType == "image/tiff" || mimeType == "image/bmp",
               let nsImage = NSImage(data: data),
               let tiffRep = nsImage.tiffRepresentation,
               let bitmapRep = NSBitmapImageRep(data: tiffRep),
               let pngData = bitmapRep.representation(using: .png, properties: [:]) {
                finalData = pngData
                finalMime = "image/png"
            } else {
                finalData = data
                finalMime = mimeType
            }

            searchViewModel?.imageAttachment = PulsarImageAttachment(
                filename: "clipboard.png",
                mimeType: finalMime,
                data: finalData
            )
        }

        private func hitToDict(_ hit: SearchHit) -> [String: Any] {
            var d: [String: Any] = [
                "id": hit.docId,
                "filename": hit.filename,
                "modality": hit.modality,
                "score": hit.scoreVector ?? hit.similarity
            ]
            if let path = hit.filePaths.first { d["path"] = path }
            if let thumb = hit.thumbnailPath { d["thumbnailPath"] = thumb }
            if let text = hit.rawText { d["rawText"] = text }
            return d
        }

        // MARK: - Push data to JS

        @MainActor private func pushResultsToJS(results: [[String: Any]], phase: String) {
            guard let webView else { return }
            let payload: [String: Any] = ["results": results, "phase": phase]
            guard let jsonData = try? JSONSerialization.data(withJSONObject: payload),
                  let jsonStr = String(data: jsonData, encoding: .utf8) else { return }
            let js = "if(typeof onSearchResults==='function')onSearchResults(\(jsonStr));"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        @MainActor private func pushErrorToJS(_ message: String) {
            guard let webView else { return }
            let escaped = message.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
            let js = "if(typeof onSearchResults==='function')onSearchResults({results:[],phase:'error',errorMessage:'\(escaped)'});"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        @MainActor private func pushHighlights(ids: [String]) {
            guard let webView else { return }
            let idsStr = ids.map { "'\($0)'" }.joined(separator: ",")
            let js = "if(typeof highlightStars==='function')highlightStars([\(idsStr)]);"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        // MARK: - Focus search (called from keyboard shortcut)

        @MainActor func focusSearchInput() {
            guard let webView else { return }
            webView.evaluateJavaScript("if(typeof focusSearch==='function')focusSearch();", completionHandler: nil)
        }

        // MARK: - Open Settings (via menu item simulation)

        /// Open Settings by finding the menu item and performing its action.
        /// `sendAction(_:to:from:)` with nil target silently fails in this
        /// app architecture (.accessory policy + SwiftUI Settings scene).
        /// The proven workaround is to locate the item in NSApp.mainMenu.
        private static func openSettingsViaMenu() {
            let knownSelectors: Set<String> = [
                "showSettingsWindow:",
                "showPreferencesWindow:",
            ]
            let knownTitles: Set<String> = [
                "Settings\u{2026}", "Settings",
                "Preferences\u{2026}", "Preferences",
            ]
            guard let mainMenu = NSApp.mainMenu else {
                NSLog("[Nebula] No main menu available for Settings")
                return
            }

            for topLevel in mainMenu.items {
                guard let submenu = topLevel.submenu else { continue }
                // Pass 1: by selector name
                for (index, item) in submenu.items.enumerated() {
                    if let action = item.action,
                       knownSelectors.contains(NSStringFromSelector(action)) {
                        submenu.performActionForItem(at: index)
                        return
                    }
                }
                // Pass 2: by title (fallback)
                for (index, item) in submenu.items.enumerated()
                where knownTitles.contains(item.title) {
                    submenu.performActionForItem(at: index)
                    return
                }
            }
            NSLog("[Nebula] Settings menu item not found")
        }

        // MARK: - Load HTML

        @MainActor func loadWithData(_ stars: [NebulaStar], in webView: WKWebView) {
            hasLoaded = true
            loadedStarCount = stars.count

            let htmlSource: URL?
            if let bundled = Bundle.main.url(forResource: "nebula", withExtension: "html", subdirectory: "nebula") {
                htmlSource = bundled
            } else if let repoRoot = Bundle.main.infoDictionary?["RubickDevRepoRoot"] as? String {
                let path = URL(fileURLWithPath: repoRoot)
                    .appendingPathComponent("apps/Rubick/Resources/nebula/nebula.html")
                htmlSource = FileManager.default.fileExists(atPath: path.path) ? path : nil
            } else {
                htmlSource = nil
            }

            guard let source = htmlSource else {
                #if DEBUG
                print("[Nebula] Cannot find nebula.html")
                #endif
                return
            }

            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("rubick-nebula-\(ProcessInfo.processInfo.processIdentifier)")
            try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

            let dataJS = buildDataJS(stars)
            let dataFile = tempDir.appendingPathComponent("data.js")
            try? dataJS.write(to: dataFile, atomically: true, encoding: .utf8)

            let htmlDest = tempDir.appendingPathComponent("nebula.html")
            try? FileManager.default.removeItem(at: htmlDest)
            try? FileManager.default.copyItem(at: source, to: htmlDest)

            // Copy module JS files for visual upgrade
            let moduleFiles = [
                "nebula-atmosphere", "nebula-colors", "nebula-clouds",
                "nebula-energy", "nebula-interactions", "nebula-guardian"
            ]
            for filename in moduleFiles {
                if let moduleSource = Bundle.main.url(
                    forResource: filename, withExtension: "js", subdirectory: "nebula"
                ) {
                    let moduleDest = tempDir.appendingPathComponent("\(filename).js")
                    try? FileManager.default.removeItem(at: moduleDest)
                    try? FileManager.default.copyItem(at: moduleSource, to: moduleDest)
                } else if let repoRoot = Bundle.main.infoDictionary?["RubickDevRepoRoot"] as? String {
                    let modulePath = URL(fileURLWithPath: repoRoot)
                        .appendingPathComponent("apps/Rubick/Resources/nebula/\(filename).js")
                    if FileManager.default.fileExists(atPath: modulePath.path) {
                        let moduleDest = tempDir.appendingPathComponent("\(filename).js")
                        try? FileManager.default.removeItem(at: moduleDest)
                        try? FileManager.default.copyItem(at: modulePath, to: moduleDest)
                    }
                }
            }

            // Copy assets directory (Rubick PNG images)
            let assetsDest = tempDir.appendingPathComponent("assets")
            try? FileManager.default.removeItem(at: assetsDest)
            if let assetsSource = Bundle.main.url(
                forResource: "assets", withExtension: nil, subdirectory: "nebula"
            ) {
                try? FileManager.default.copyItem(at: assetsSource, to: assetsDest)
            } else if let repoRoot = Bundle.main.infoDictionary?["RubickDevRepoRoot"] as? String {
                let assetsPath = URL(fileURLWithPath: repoRoot)
                    .appendingPathComponent("apps/Rubick/Resources/nebula/assets")
                if FileManager.default.fileExists(atPath: assetsPath.path) {
                    try? FileManager.default.copyItem(at: assetsPath, to: assetsDest)
                }
            }

            // Copy lib directory (Three.js + extensions, bundled locally)
            let libDest = tempDir.appendingPathComponent("lib")
            try? FileManager.default.removeItem(at: libDest)
            if let libSource = Bundle.main.url(
                forResource: "lib", withExtension: nil, subdirectory: "nebula"
            ) {
                try? FileManager.default.copyItem(at: libSource, to: libDest)
            } else if let repoRoot = Bundle.main.infoDictionary?["RubickDevRepoRoot"] as? String {
                let libPath = URL(fileURLWithPath: repoRoot)
                    .appendingPathComponent("apps/Rubick/Resources/nebula/lib")
                if FileManager.default.fileExists(atPath: libPath.path) {
                    try? FileManager.default.copyItem(at: libPath, to: libDest)
                }
            }

            let accessRoot = URL(fileURLWithPath: "/")
            webView.loadFileURL(htmlDest, allowingReadAccessTo: accessRoot)
        }

        private func buildDataJS(_ stars: [NebulaStar]) -> String {
            var js = "const NEBULA_DATA = ["
            for (i, star) in stars.enumerated() {
                if i > 0 { js += "," }
                let m = star.modality == "video" ? "v" : "i"
                let thumb = star.thumbnailPath ?? ""
                let escapedThumb = thumb.replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "'", with: "\\'")
                let escapedId = star.id.replacingOccurrences(of: "'", with: "\\'")
                let escapedFilename = star.filename
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "'", with: "\\'")
                js += "{id:'\(escapedId)',x:\(star.x),y:\(star.y),z:\(star.z),m:'\(m)',c:\(star.cluster),t:'\(escapedThumb)',f:'\(escapedFilename)'}"
            }
            js += "];"
            return js
        }
    }
}
