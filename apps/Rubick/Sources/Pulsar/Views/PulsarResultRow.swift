import SwiftUI

/// Single search result row.
///
/// v4 visual spec:
/// - 44×44 thumbnail (image: actual thumbnail/gradient placeholder,
///   text: text lines, video: frame capture/play button)
/// - Two text lines (filename + parent directory path); selected row
///   shows a third snippet line
/// - Selected row: gradient background + left 2.5 px glow bar +
///   thumbnail bloom glow
/// - Right side: score pill (scoreVector) when selected; always shows
///   Cmd+N ghost badge
struct PulsarResultRow: View {

    let hit: SearchHit
    let index: Int           // 0-based, used for Cmd+N badge
    let isSelected: Bool

    var body: some View {
        ZStack(alignment: .leading) {
            // Selected row background gradient
            if isSelected {
                selectedBackground
                selectedLeftBar
            }

            // Row content
            HStack(spacing: 13) {
                thumbnail
                textStack
                Spacer(minLength: 0)
                metaStack
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 9)
        }
        .contentShape(Rectangle())
    }

    // MARK: - Background

    private var selectedBackground: some View {
        PulsarColors.selectedRowFill
    }

    private var selectedLeftBar: some View {
        Rectangle()
            .fill(
                LinearGradient(
                    colors: [Color(hex: "5eead4"), Color(hex: "22d3ee")],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .frame(width: 2.5)
            .padding(.vertical, 5)
            .shadow(color: Color(hex: "5eead4").opacity(0.9), radius: 4, x: 0, y: 0)
            .shadow(color: Color(hex: "5eead4").opacity(0.4), radius: 10, x: 0, y: 0)
            .shadow(color: Color(hex: "22d3ee").opacity(0.2), radius: 20, x: 0, y: 0)
            .frame(maxHeight: .infinity, alignment: .leading)
    }

    // MARK: - Thumbnail

    @ViewBuilder
    private var thumbnail: some View {
        ZStack {
            thumbnailBackground
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.white.opacity(isSelected ? 0.15 : 0.10), lineWidth: 0.5)
                )
                // Bloom glow on selection
                .shadow(color: Color.black.opacity(0.5), radius: isSelected ? 4 : 3, x: 0, y: 2)
                .shadow(color: isSelected ? Color(hex: "14b8a6").opacity(0.15) : .clear, radius: 8, x: 0, y: 0)
                .shadow(color: isSelected ? Color(hex: "5eead4").opacity(0.25) : .clear, radius: 16, x: 0, y: 0)
                .shadow(color: isSelected ? Color(hex: "22d3ee").opacity(0.10) : .clear, radius: 32, x: 0, y: 0)
        }
    }

    @ViewBuilder
    private var thumbnailBackground: some View {
        let modality = Modality(rawString: hit.modality)
        switch modality {
        case .image:
            if let path = hit.thumbnailPath, let img = NSImage(contentsOfFile: path) {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFill()
                    .clipped()
            } else {
                // Gradient placeholder when no thumbnail available
                ZStack {
                    LinearGradient(
                        colors: thumbnailGradientColors(for: index),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    LinearGradient(
                        colors: [Color.white.opacity(0.12), Color.clear],
                        startPoint: .topLeading,
                        endPoint: .center
                    )
                }
            }

        case .video:
            ZStack {
                if let path = hit.thumbnailPath, let img = NSImage(contentsOfFile: path) {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFill()
                        .clipped()
                        .overlay(
                            ZStack {
                                Color.black.opacity(0.28)
                                Image(systemName: "play.fill")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundStyle(Color.white.opacity(0.90))
                                    .shadow(color: Color.black.opacity(0.6), radius: 3)
                                    .offset(x: 1)
                            }
                        )
                } else {
                    LinearGradient(
                        colors: [Color(hex: "042f2e"), Color(hex: "115e59"), Color(hex: "14b8a6")],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    Image(systemName: "play.fill")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(Color.white.opacity(0.85))
                        .shadow(color: Color(hex: "5eead4").opacity(0.6), radius: 4)
                        .offset(x: 1)
                    LinearGradient(
                        colors: [Color.white.opacity(0.12), Color.clear],
                        startPoint: .topLeading,
                        endPoint: .center
                    )
                }
            }

        default: // text, unknown
            ZStack {
                LinearGradient(
                    colors: [Color(hex: "0f2820"), Color(hex: "1a5c3a"), Color(hex: "2d8a5a")],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                VStack(spacing: 3) {
                    ForEach([1.0, 0.65, 0.55, 0.35], id: \.self) { opacity in
                        RoundedRectangle(cornerRadius: 1)
                            .fill(Color(hex: "6ee7b7").opacity(opacity))
                            .frame(
                                width: opacity == 1.0 ? 22 : opacity == 0.65 ? 18 : opacity == 0.55 ? 20 : 13,
                                height: 2.5
                            )
                    }
                }
                LinearGradient(
                    colors: [Color.white.opacity(0.12), Color.clear],
                    startPoint: .topLeading,
                    endPoint: .center
                )
            }
        }
    }

    /// Return distinct gradient colours per result index for image placeholders.
    private func thumbnailGradientColors(for idx: Int) -> [Color] {
        let palettes: [[Color]] = [
            [Color(hex: "042f2e"), Color(hex: "115e59"), Color(hex: "14b8a6")],
            [Color(hex: "1a3a6a"), Color(hex: "3b6ea8"), Color(hex: "8ab8e8")],
            [Color(hex: "051a2e"), Color(hex: "0a3a5e"), Color(hex: "1a6a9a")],
            [Color(hex: "022c22"), Color(hex: "065f46"), Color(hex: "10b981")],
            [Color(hex: "0a1a2a"), Color(hex: "164e63"), Color(hex: "22d3ee")],
        ]
        return palettes[idx % palettes.count]
    }

    // MARK: - Text

    private var textStack: some View {
        VStack(alignment: .leading, spacing: 2) {
            // Filename
            Text(hit.filename)
                .font(.system(size: 13.5, weight: .medium))
                .foregroundStyle(isSelected ? Color.white.opacity(0.96) : Color.white.opacity(0.70))
                .tracking(-0.012 * 13.5)
                .lineLimit(1)
                .truncationMode(.middle)

            // Parent directory path
            Text(shortPath)
                .font(.system(size: 11))
                .foregroundStyle(Color.white.opacity(0.26))
                .lineLimit(1)
                .truncationMode(.head)

            // Snippet line: shown only when selected
            if isSelected, let snippet = snippetText {
                snippetView(snippet)
            }
        }
    }

    @ViewBuilder
    private func snippetView(_ text: String) -> some View {
        // For images: format is "4032 × 3024 · 6.2 MB · Jun 14, 2024"
        if Modality(rawString: hit.modality) == .image, let parts = parseImageSnippet(text) {
            let attrStr = buildImageSnippet(parts)
            Text(attrStr)
                .font(.system(size: 11))
                .lineLimit(1)
                .truncationMode(.tail)
        } else {
            Text(text)
                .font(.system(size: 11).italic())
                .foregroundStyle(Color.white.opacity(0.30))
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    /// Parse image metadata string into components: dimensions · size · date.
    private func parseImageSnippet(_ text: String) -> (dims: String, size: String, date: String?)? {
        let parts = text.components(separatedBy: " · ")
        guard parts.count >= 2 else { return nil }
        return (dims: parts[0], size: parts[1], date: parts.count >= 3 ? parts[2] : nil)
    }

    private func buildImageSnippet(_ parts: (dims: String, size: String, date: String?)) -> AttributedString {
        func dimPart(_ text: String) -> AttributedString {
            var a = AttributedString(text)
            a.swiftUI.font = .system(size: 11).italic()
            a.swiftUI.foregroundColor = Color.white.opacity(0.30)
            return a
        }
        func sepPart() -> AttributedString {
            var a = AttributedString(" · ")
            a.swiftUI.font = .system(size: 11).italic()
            a.swiftUI.foregroundColor = Color.white.opacity(0.20)
            return a
        }

        var result = dimPart(parts.dims)
        result += sepPart()

        // File size: teal highlight
        var size = AttributedString(parts.size)
        size.swiftUI.font = .system(size: 11, weight: .medium)
        size.swiftUI.foregroundColor = Color(red: 94/255, green: 234/255, blue: 212/255).opacity(0.85)
        result += size

        if let date = parts.date {
            result += sepPart()
            result += dimPart(date)
        }

        return result
    }

    /// Short path: show only the parent directory.
    private var shortPath: String {
        let path = hit.filePaths.first ?? ""
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let display = path.hasPrefix(home)
            ? "~" + path.dropFirst(home.count)
            : path
        let url = URL(fileURLWithPath: display)
        return url.deletingLastPathComponent().path
    }

    /// Snippet text:
    /// - image: file metadata → "4032 × 3024 · 6.2 MB · Jun 14, 2024"
    /// - video/text: rawText excerpt
    private var snippetText: String? {
        let modality = Modality(rawString: hit.modality)
        switch modality {
        case .image:
            return imageMetaSnippet
        case .video:
            return hit.rawText.flatMap { $0.isEmpty ? nil : String($0.prefix(80)) }
        default:
            return hit.rawText.flatMap { $0.isEmpty ? nil : String($0.prefix(80)) }
        }
    }

    /// Read actual file metadata from disk: dimensions + file size + modification date.
    private var imageMetaSnippet: String? {
        guard let path = hit.filePaths.first else { return nil }
        let url = URL(fileURLWithPath: path)

        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let fileSize = attrs?[.size] as? Int64 ?? 0
        let modDate = attrs?[.modificationDate] as? Date

        let sizeFmt = ByteCountFormatter()
        sizeFmt.allowedUnits = [.useKB, .useMB, .useGB]
        sizeFmt.countStyle = .file
        let sizeStr = sizeFmt.string(fromByteCount: fileSize)

        var dateStr: String?
        if let modDate {
            let df = DateFormatter()
            df.dateStyle = .medium
            df.timeStyle = .none
            dateStr = df.string(from: modDate)
        }

        // Read image dimensions (prefer original file via CGImageSource for speed)
        var dimsStr: String?
        if let src = CGImageSourceCreateWithURL(url as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any],
           let w = props[kCGImagePropertyPixelWidth as String] as? Int,
           let h = props[kCGImagePropertyPixelHeight as String] as? Int {
            dimsStr = "\(w) × \(h)"
        } else if let img = NSImage(contentsOfFile: path) {
            let sz = img.size
            dimsStr = "\(Int(sz.width)) × \(Int(sz.height))"
        }

        var parts: [String] = []
        if let d = dimsStr { parts.append(d) }
        if fileSize > 0 { parts.append(sizeStr) }
        if let s = dateStr { parts.append(s) }

        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: - Right-side meta

    private var metaStack: some View {
        VStack(alignment: .trailing, spacing: 5) {
            if isSelected {
                scorePill
            }
            cmdBadge
        }
    }

    private var scorePill: some View {
        let score = hit.scoreVector ?? hit.similarity
        return Text(String(format: "%.2f", score))
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(Color(hex: "a7f3d0").opacity(0.95))
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(
                PulsarColors.scorePillFill
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(hex: "5eead4").opacity(0.35), lineWidth: 0.5)
            )
            .cornerRadius(6)
            .shadow(color: Color(hex: "5eead4").opacity(0.12), radius: 4)
    }

    private var cmdBadge: some View {
        Text("⌘\(index + 1)")
            .font(.system(size: 9, design: .monospaced))
            .foregroundStyle(Color.white.opacity(0.10))
            .tracking(0.02 * 9)
    }
}
