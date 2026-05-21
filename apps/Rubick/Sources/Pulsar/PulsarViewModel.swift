import AppKit
import Combine
import Foundation

/// Lightweight image attachment for Pulsar's clipboard-paste search.
struct PulsarImageAttachment: Equatable {
    let filename: String
    let mimeType: String
    let data: Data

    var formattedSize: String {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useKB, .useMB]
        f.countStyle = .file
        return f.string(fromByteCount: Int64(data.count))
    }
}

/// Pulsar's state machine.
///
/// Search flow:
/// 1. Text: user presses Enter → ``submitSearch()`` → 300 ms debounce → request
/// 2. Image paste (⌘V): ``pastedImage`` changes → same debounce path
/// 3. ``performSearch`` calls ``client.search`` or ``searchFused`` with Settings top-K
/// 5. `selectedIndex` resets to 0 on every new result set
@MainActor
final class PulsarViewModel: ObservableObject {

    // MARK: - Published state

    @Published var query: String = "" {
        didSet {
            // Only clear results when query is emptied; actual search triggers on Enter
            if query.trimmingCharacters(in: .whitespaces).isEmpty && pastedImage == nil {
                debounceTask?.cancel()
                results = []
                phase = .empty
                selectedIndex = 0
                viewportStart = 0
            }
        }
    }
    @Published private(set) var results: [SearchHit] = []
    @Published private(set) var phase: Phase = .empty
    @Published private(set) var selectedIndex: Int = 0
    /// Total hit count from the backend (may exceed the fetched limit).
    @Published private(set) var totalCount: Int = 0

    /// Pasted image for image-only search (Cmd+V from clipboard).
    @Published var pastedImage: PulsarImageAttachment? {
        didSet { scheduleSearch() }
    }

    /// Start index of the visible viewport — determines which 5 rows
    /// are currently shown.
    @Published private(set) var viewportStart: Int = 0

    /// Maximum number of rows visible in the Pulsar viewport at once.
    static let viewportSize = 5

    /// The results currently visible in the viewport (up to 5).
    var visibleResults: [SearchHit] {
        guard !results.isEmpty else { return [] }
        let end = min(viewportStart + Self.viewportSize, results.count)
        return Array(results[viewportStart..<end])
    }

    /// The selected index relative to the current viewport (for highlighting).
    var selectedIndexInViewport: Int {
        selectedIndex - viewportStart
    }

    // MARK: - Phase

    enum Phase: Equatable {
        case empty       // query is blank — show only the search bar
        case loading     // debounce fired, request in flight — skeleton
        case results     // have results
        case noResults   // non-empty query but 0 results
        case error(String)

        static func == (lhs: Phase, rhs: Phase) -> Bool {
            switch (lhs, rhs) {
            case (.empty, .empty), (.loading, .loading),
                 (.results, .results), (.noResults, .noResults): return true
            case (.error(let a), .error(let b)): return a == b
            default: return false
            }
        }
    }

    // MARK: - Dependencies

    /// Injected by PulsarWindowController before `show()`.
    var client: RubickClient?

    // MARK: - Private

    private var debounceTask: Task<Void, Never>?
    /// Tracks the query text that produced the current result set.
    private var lastSearchedQuery: String = ""

    // MARK: - Public actions

    func moveSelection(by delta: Int) {
        guard !results.isEmpty else { return }
        let newIndex = (selectedIndex + delta + results.count) % results.count
        selectedIndex = newIndex
        adjustViewport()
    }

    /// Select a specific index directly (tap on row).
    func selectIndex(_ idx: Int) {
        guard idx < results.count else { return }
        selectedIndex = idx
        adjustViewport()
    }

    /// Select and open (used for Cmd+1-5 shortcuts — relative to viewport).
    func selectAndOpen(at viewportRelativeIdx: Int) {
        let absoluteIdx = viewportStart + viewportRelativeIdx
        guard absoluteIdx < results.count else { return }
        selectedIndex = absoluteIdx
    }

    var selectedHit: SearchHit? {
        guard selectedIndex < results.count else { return nil }
        return results[selectedIndex]
    }

    func clearQuery() {
        debounceTask?.cancel()
        query = ""
        pastedImage = nil
        results = []
        phase = .empty
        selectedIndex = 0
        totalCount = 0
        viewportStart = 0
        lastSearchedQuery = ""
    }

    // MARK: - Viewport adjustment

    /// Ensure `selectedIndex` is always within the visible viewport
    /// window [viewportStart, viewportStart + viewportSize).
    private func adjustViewport() {
        if selectedIndex < viewportStart {
            viewportStart = selectedIndex
        } else if selectedIndex >= viewportStart + Self.viewportSize {
            viewportStart = selectedIndex - Self.viewportSize + 1
        }
    }

    // MARK: - Search flow

    /// Trigger search explicitly (called on Enter key press).
    func submitSearch() {
        lastSearchedQuery = query.trimmingCharacters(in: .whitespaces)
        scheduleSearch()
    }

    /// Whether the current query differs from what produced the visible results.
    var queryChanged: Bool {
        query.trimmingCharacters(in: .whitespaces) != lastSearchedQuery
    }

    private func scheduleSearch() {
        debounceTask?.cancel()

        let trimmed = query.trimmingCharacters(in: .whitespaces)
        let hasImage = pastedImage != nil
        guard !trimmed.isEmpty || hasImage else {
            results = []
            phase = .empty
            selectedIndex = 0
            viewportStart = 0
            return
        }

        debounceTask = Task {
            // 300 ms debounce
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }

            phase = .loading

            await performSearch(query: trimmed)
        }
    }

    private func performSearch(query: String) async {
        guard let client else {
            phase = .error("Backend not ready")
            return
        }

        // Read user-configured topK (default 20, shared with Nebula)
        let stored = UserDefaults.standard.integer(forKey: "search_top_k")
        let topK = min(50, max(1, stored == 0 ? 20 : stored))

        do {
            let response: SearchResponse
            if let img = pastedImage {
                // Image-only or image+text fused search
                response = try await client.searchFused(
                    q: query,
                    imageData: img.data,
                    imageFilename: img.filename,
                    imageMimeType: img.mimeType,
                    textWeight: query.isEmpty ? 0.0 : 0.5,
                    limit: topK
                )
            } else {
                response = try await client.search(q: query, limit: topK)
            }
            guard !Task.isCancelled else { return }

            results = response.results
            selectedIndex = 0
            viewportStart = 0
            totalCount = response.count

            if response.results.isEmpty {
                phase = .noResults
            } else {
                phase = .results
            }
        } catch {
            guard !Task.isCancelled else { return }
            phase = .error(error.localizedDescription)
        }
    }

    /// Paste image from the system clipboard. Returns true if an image was found.
    @discardableResult
    func pasteFromClipboard() -> Bool {
        let pb = NSPasteboard.general
        guard let imageData = pb.data(forType: .tiff) ?? pb.data(forType: .png) else {
            return false
        }
        // Determine MIME type
        let mimeType: String
        if pb.data(forType: .png) != nil {
            mimeType = "image/png"
        } else {
            mimeType = "image/tiff"
        }
        // Convert TIFF to PNG for the backend (it expects standard formats)
        let finalData: Data
        let finalMime: String
        if mimeType == "image/tiff",
           let nsImage = NSImage(data: imageData),
           let tiffRep = nsImage.tiffRepresentation,
           let bitmapRep = NSBitmapImageRep(data: tiffRep),
           let pngData = bitmapRep.representation(using: .png, properties: [:]) {
            finalData = pngData
            finalMime = "image/png"
        } else {
            finalData = imageData
            finalMime = mimeType
        }

        pastedImage = PulsarImageAttachment(
            filename: "clipboard.png",
            mimeType: finalMime,
            data: finalData
        )
        return true
    }
}
