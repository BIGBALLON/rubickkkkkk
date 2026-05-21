import Combine
import Foundation
import SwiftUI

/// Persistent list of folders the user wants Rubick to watch.
///
/// State machine for a single folder:
///
///     idle → queued (POST /index/job) → indexing → ready
///            ↘ failed (user can retry from the row)
///
/// Per folder we keep the URL itself (canonicalized, no trailing
/// slash) plus a security-scoped bookmark so the path keeps working
/// across launches even if the user moved or renamed an ancestor
/// directory. The bookmark also future-proofs us for when we turn
/// on the macOS App Sandbox (currently off — see ``Project.swift``).
///
/// Persistence: JSON file at
///     ``~/Library/Application Support/Rubick/watched_folders.json``
///
/// Format:
///     [{ "bookmark": "<base64>", "path": "<abs>" }, …]
///
/// We store ``path`` redundantly for diagnostics — bookmark decoding
/// can fail (e.g. drive unmounted) and we want to show the user a
/// readable error rather than a blank entry.
@MainActor
final class WatchedFoldersStore: ObservableObject {
    @Published private(set) var folders: [WatchedFolder] = []

    /// File URL where the folder list is persisted.
    private static var storageURL: URL {
        let root = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!.appendingPathComponent("Rubick")
        return root.appendingPathComponent("watched_folders.json")
    }

    init() {
        self.folders = Self.load()
        // If we migrated from UserDefaults, persist to new location immediately
        if !folders.isEmpty && !FileManager.default.fileExists(atPath: Self.storageURL.path) {
            persist()
        }
    }

    // MARK: - Mutations

    /// Add ``url`` to the watch list. Returns ``false`` if it was
    /// already present (deduplicated by canonical path).
    @discardableResult
    func add(_ url: URL) -> Bool {
        let canon = url.standardized
        if folders.contains(where: { $0.url.standardized == canon }) {
            return false
        }
        let bookmark = try? canon.bookmarkData(
            options: [.minimalBookmark],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        folders.append(WatchedFolder(url: canon, bookmark: bookmark))
        persist()
        return true
    }

    func remove(_ folder: WatchedFolder) {
        folders.removeAll { $0.id == folder.id }
        persist()
    }

    /// Refresh per-folder transient state (status / last indexed-at).
    /// Persisted only for human readability — the source of truth
    /// for "is this folder fully indexed?" is the LanceDB rows we
    /// look up via /search at query time.
    func setStatus(_ status: WatchedFolder.Status, for folder: WatchedFolder) {
        guard let i = folders.firstIndex(where: { $0.id == folder.id }) else { return }
        folders[i].status = status
        persist()
    }

    // MARK: - Persistence (file-based)

    private func persist() {
        let url = Self.storageURL
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(folders.map(WatchedFolderPersist.init)) else {
            FileHandle.standardError.write(
                Data("[WatchedFoldersStore] encode failed\n".utf8)
            )
            return
        }
        // Atomic write via temp file
        let tmp = url.appendingPathExtension("tmp")
        do {
            try data.write(to: tmp, options: .atomic)
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } catch {
            // replaceItemAt fails if destination doesn't exist yet
            try? FileManager.default.moveItem(at: tmp, to: url)
        }
    }

    private static func load() -> [WatchedFolder] {
        let url = storageURL

        // 1. Try file-based storage (current format)
        if let data = try? Data(contentsOf: url),
           let stored = try? JSONDecoder().decode([WatchedFolderPersist].self, from: data) {
            return resolveBookmarks(stored)
        }

        // 2. One-time migration from UserDefaults (pre-0.0.5 format)
        if let legacyData = UserDefaults.standard.data(forKey: "watched_folders"),
           let stored = try? JSONDecoder().decode([LegacyWatchedFolderPersist].self, from: legacyData) {
            let folders = resolveLegacyBookmarks(stored)
            // Remove old key — data will be persisted to file in init()
            UserDefaults.standard.removeObject(forKey: "watched_folders")
            return folders
        }

        return []
    }

    /// Resolve bookmarks from the current file format (base64 strings).
    private static func resolveBookmarks(_ stored: [WatchedFolderPersist]) -> [WatchedFolder] {
        stored.compactMap { p -> WatchedFolder? in
            if let bookmarkData = p.bookmarkData {
                var stale = false
                if let resolved = try? URL(
                    resolvingBookmarkData: bookmarkData,
                    options: [.withoutUI],
                    relativeTo: nil,
                    bookmarkDataIsStale: &stale
                ) {
                    return WatchedFolder(url: resolved, bookmark: bookmarkData)
                }
            }
            // Fallback: use literal path
            return WatchedFolder(
                url: URL(fileURLWithPath: p.path),
                bookmark: p.bookmarkData
            )
        }
    }

    /// Resolve bookmarks from the legacy UserDefaults format (raw Data).
    private static func resolveLegacyBookmarks(_ stored: [LegacyWatchedFolderPersist]) -> [WatchedFolder] {
        stored.compactMap { p -> WatchedFolder? in
            if let bookmark = p.bookmark {
                var stale = false
                if let resolved = try? URL(
                    resolvingBookmarkData: bookmark,
                    options: [.withoutUI],
                    relativeTo: nil,
                    bookmarkDataIsStale: &stale
                ) {
                    return WatchedFolder(url: resolved, bookmark: bookmark)
                }
            }
            return WatchedFolder(
                url: URL(fileURLWithPath: p.path),
                bookmark: p.bookmark
            )
        }
    }
}

// MARK: - Models

/// One row in the watched-folders list. Identity is the canonicalized
/// path string so we can route status updates per-folder reliably
/// across persistence reloads.
struct WatchedFolder: Identifiable, Sendable, Hashable {
    var url: URL
    var bookmark: Data?
    var status: Status = .idle

    var id: String { url.standardized.path }
    var displayName: String { url.lastPathComponent }
    var displayPath: String { url.standardized.path }

    func hash(into hasher: inout Hasher) {
        // Identity = canonical path. Two ``WatchedFolder`` values that
        // point at the same directory hash equal even if their
        // bookmarks / transient ``status`` diverge.
        hasher.combine(id)
    }

    static func == (lhs: WatchedFolder, rhs: WatchedFolder) -> Bool {
        lhs.id == rhs.id
    }

    enum Status: Sendable, Equatable {
        case idle
        /// Live ingest. ``progress`` is published by ``WatchService.kick``
        /// from each ``GET /index/job/{id}`` poll so the sidebar card
        /// + ``IngestProgressBanner`` can render a real bar instead of
        /// an indeterminate spinner. ``nil`` while the job is queued
        /// (no walk has happened yet) or when polling against an old
        /// backend that doesn't emit the field.
        case indexing(jobId: String, progress: IndexJob.Progress?)
        case ready(at: Date, stats: IndexJob.Stats)
        case failed(reason: String)

        var label: String {
            switch self {
            case .idle: return "Idle"
            case .indexing: return "Indexing…"
            case .ready: return "Indexed"
            case .failed: return "Failed"
            }
        }

        var tint: Color {
            switch self {
            case .idle: return .secondary
            case .indexing: return .yellow
            case .ready: return .green
            case .failed: return .red
            }
        }
    }
}

// MARK: - Codable persistence (current format — base64 bookmark)

/// Stored in ``watched_folders.json``. Bookmark is base64-encoded
/// for human readability when inspecting the file.
private struct WatchedFolderPersist: Codable {
    let path: String
    let bookmark: String? // base64-encoded bookmark data

    init(_ f: WatchedFolder) {
        self.path = f.url.standardized.path
        self.bookmark = f.bookmark?.base64EncodedString()
    }

    var bookmarkData: Data? {
        guard let b = bookmark else { return nil }
        return Data(base64Encoded: b)
    }
}

// MARK: - Legacy format (UserDefaults migration)

/// The pre-0.0.5 format stored raw ``Data`` for bookmarks in
/// UserDefaults. This struct exists solely for one-time migration.
private struct LegacyWatchedFolderPersist: Codable {
    let path: String
    let bookmark: Data?
}
