// apps/Rubick/Sources/Nebula/NebulaSearchViewModel.swift
import AppKit
import Combine
import Foundation

/// Search state for the Nebula main window.
///
/// Manages query input, debounced search, facet filters, and
/// result list. Communicates matched star IDs back to NebulaViewModel
/// for 3D highlighting.
@MainActor
final class NebulaSearchViewModel: ObservableObject {

    // MARK: - Published state

    @Published var query: String = "" {
        didSet {
            // Only clear when query is emptied; search triggers on Enter via submitSearch()
            if query.trimmingCharacters(in: .whitespaces).isEmpty && imageAttachment == nil {
                debounceTask?.cancel()
                results = []
                phase = .idle
                selectedIndex = 0
                matchedStarIds = []
            }
        }
    }
    @Published private(set) var results: [SearchHit] = []
    @Published private(set) var phase: Phase = .idle
    @Published var selectedIndex: Int = 0

    /// Facet: modality filter. Empty = no filter.
    @Published var modalityFilter: Set<String> = [] {
        didSet { rerunIfActive() }
    }
    /// Facet: time range. nil = no bound.
    @Published var mtimeAfter: Date? {
        didSet { rerunIfActive() }
    }
    @Published var mtimeBefore: Date? {
        didSet { rerunIfActive() }
    }
    /// Facet: path prefix filter. nil = no folder filter.
    @Published var pathPrefix: String? {
        didSet { rerunIfActive() }
    }
    /// Image attachment for fused search.
    /// Setting to nil clears fused state; actual search triggers on Enter via submitSearch().
    @Published var imageAttachment: PulsarImageAttachment?

    /// IDs of stars that match the current search (for 3D highlighting).
    @Published private(set) var matchedStarIds: Set<String> = []

    /// Whether the result panel should be visible.
    var showResultPanel: Bool {
        switch phase {
        case .idle: return false
        case .searching, .results, .noResults, .error: return true
        }
    }

    /// Whether there's an active query (text or image).
    var hasActiveQuery: Bool {
        !query.trimmingCharacters(in: .whitespaces).isEmpty || imageAttachment != nil
    }

    // MARK: - Phase

    enum Phase: Equatable {
        case idle
        case searching
        case results
        case noResults
        case error(String)

        static func == (lhs: Phase, rhs: Phase) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.searching, .searching),
                 (.results, .results), (.noResults, .noResults): return true
            case (.error(let a), .error(let b)): return a == b
            default: return false
            }
        }
    }

    // MARK: - Dependencies

    var client: RubickClient?

    /// Reference to the Nebula map data for matching star IDs to results.
    weak var nebulaViewModel: NebulaViewModel?

    // MARK: - Private

    private var debounceTask: Task<Void, Never>?

    // MARK: - Public actions

    func moveSelection(by delta: Int) {
        guard !results.isEmpty else { return }
        selectedIndex = (selectedIndex + delta + results.count) % results.count
    }

    var selectedHit: SearchHit? {
        guard selectedIndex < results.count else { return nil }
        return results[selectedIndex]
    }

    func clearSearch() {
        debounceTask?.cancel()
        query = ""
        imageAttachment = nil
        results = []
        phase = .idle
        selectedIndex = 0
        matchedStarIds = []
    }

    // MARK: - Search flow

    /// Trigger search explicitly (called when user presses Enter or facets change).
    func submitSearch() {
        scheduleSearch()
    }

    private func rerunIfActive() {
        guard hasActiveQuery, phase != .idle else { return }
        scheduleSearch()
    }

    private func scheduleSearch() {
        debounceTask?.cancel()

        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty || imageAttachment != nil else {
            results = []
            phase = .idle
            selectedIndex = 0
            matchedStarIds = []
            return
        }

        debounceTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            phase = .searching
            await performSearch(query: trimmed)
        }
    }

    private func performSearch(query: String) async {
        guard let client else {
            phase = .error("Backend not ready")
            return
        }

        let stored = UserDefaults.standard.integer(forKey: "search_top_k")
        let topK = min(50, max(1, stored == 0 ? 20 : stored))

        let modalities: Set<String>? = modalityFilter.isEmpty ? nil : modalityFilter

        do {
            let response: SearchResponse
            if let img = imageAttachment {
                response = try await client.searchFused(
                    q: query,
                    imageData: img.data,
                    imageFilename: img.filename,
                    imageMimeType: img.mimeType,
                    textWeight: query.isEmpty ? 0.0 : 0.5,
                    limit: topK,
                    modalities: modalities,
                    pathPrefix: pathPrefix,
                    mtimeAfter: mtimeAfter.map { Int($0.timeIntervalSince1970) },
                    mtimeBefore: mtimeBefore.map { Int($0.timeIntervalSince1970) }
                )
            } else {
                response = try await client.search(
                    q: query,
                    limit: topK,
                    modalities: modalities,
                    pathPrefix: pathPrefix,
                    mtimeAfter: mtimeAfter.map { Int($0.timeIntervalSince1970) },
                    mtimeBefore: mtimeBefore.map { Int($0.timeIntervalSince1970) }
                )
            }
            guard !Task.isCancelled else { return }

            results = response.results
            selectedIndex = 0
            phase = response.results.isEmpty ? .noResults : .results

            // Match results to star IDs for 3D highlighting
            updateMatchedStars()
        } catch {
            guard !Task.isCancelled else { return }
            phase = .error(error.localizedDescription)
        }
    }

    /// Cross-reference search results with Nebula map stars by doc_id.
    private func updateMatchedStars() {
        guard let stars = nebulaViewModel?.stars else {
            matchedStarIds = []
            return
        }
        let resultDocIds = Set(results.compactMap { $0.docId })
        matchedStarIds = Set(stars.filter { resultDocIds.contains($0.docId) }.map(\.id))
    }
}
