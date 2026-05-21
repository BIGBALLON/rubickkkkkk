import Combine
import Foundation

/// Star data from the backend map response.
struct NebulaStar: Identifiable, Equatable {
    let id: String          // chunk_id
    let docId: String
    var x: Float            // [0, 1]
    var y: Float            // [0, 1]
    var z: Float            // [0, 1]
    let cluster: Int        // HDBSCAN cluster label
    let modality: String    // "image" | "video"
    let thumbnailPath: String?
    let filename: String
}

/// Central state for the Nebula view.
///
/// Manages backend communication (map loading, recompute triggering).
/// All rendering and interaction is handled by the Three.js WebView.
@MainActor
final class NebulaViewModel: ObservableObject {
    // MARK: - Map data

    @Published var stars: [NebulaStar] = []
    @Published var mapComputedAt: Date?
    @Published var isMapStale: Bool = false
    @Published var isMapEmpty: Bool = true
    @Published var isRecomputing: Bool = false

    // MARK: - Private

    private var client: RubickClient?

    // MARK: - Init

    init(client: RubickClient? = nil) {
        self.client = client
    }

    // MARK: - Map loading

    func loadMap() async {
        guard let client else { return }
        do {
            let response = try await client.nebulaMap()
            let newStars = response.points.map { point in
                NebulaStar(
                    id: point.chunkId,
                    docId: point.docId,
                    x: point.x,
                    y: point.y,
                    z: point.z,
                    cluster: point.cluster ?? 0,
                    modality: point.modality,
                    thumbnailPath: point.thumbnailPath,
                    filename: point.filename
                )
            }
            #if DEBUG
            print("[Nebula] loadMap: got \(newStars.count) stars")
            #endif
            self.stars = newStars
            self.isMapEmpty = newStars.isEmpty
            if response.computedAt > 0 {
                self.mapComputedAt = Date(timeIntervalSince1970: TimeInterval(response.computedAt))
            }
        } catch {
            #if DEBUG
            print("[Nebula] loadMap failed: \(error)")
            #endif
        }
    }

    func checkStatus() async {
        guard let client else { return }
        do {
            let status = try await client.nebulaStatus()
            self.isMapStale = status.stale
            self.isRecomputing = status.state == "computing"
        } catch {
            #if DEBUG
            print("NebulaViewModel.checkStatus failed: \(error)")
            #endif
        }
    }

    // MARK: - Recompute

    func triggerRecompute() async {
        guard let client else { return }
        self.isRecomputing = true
        do {
            _ = try await client.nebulaRecompute()
            // Poll status with timeout (max 60s)
            var attempts = 0
            while isRecomputing && attempts < 30 {
                try await Task.sleep(nanoseconds: 2_000_000_000)
                attempts += 1
                let status = try await client.nebulaStatus()
                if status.state == "idle" {
                    self.isRecomputing = false
                    await loadMap()
                }
            }
            // Timeout: force-load whatever map exists
            if isRecomputing {
                self.isRecomputing = false
                await loadMap()
            }
        } catch {
            self.isRecomputing = false
            // Still try to load map — backend may have partial results
            await loadMap()
            #if DEBUG
            print("NebulaViewModel.triggerRecompute failed: \(error)")
            #endif
        }
    }
}
