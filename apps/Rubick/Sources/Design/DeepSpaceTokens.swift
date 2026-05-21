// apps/Rubick/Sources/Design/DeepSpaceTokens.swift
import SwiftUI

/// Deep Space — Rubick's unified visual language.
///
/// One palette shared by Pulsar and Nebula. UI chrome is maximally quiet;
/// the 3D content is the visual hero. Teal/cyan as the singular accent
/// across both modes — the colour of starlight and nebula gas.
enum DS {
    // MARK: - Backgrounds
    static let bgDeep   = Color(red: 2/255, green: 4/255, blue: 8/255)      // #020408
    static let bgMid    = Color(red: 4/255, green: 8/255, blue: 16/255)     // #040810
    static let bgPanel  = Color.white.opacity(0.03)
    static let bgPanelHover = Color.white.opacity(0.06)

    // MARK: - Accent (teal — matches Nebula star field)
    static let accent       = Color(red: 94/255, green: 234/255, blue: 212/255)  // #5eead4
    static let accentDeep   = Color(red: 20/255, green: 184/255, blue: 166/255)  // #14b8a6
    static let accentSubtle = Color(red: 94/255, green: 234/255, blue: 212/255).opacity(0.10)
    static let accentGlow   = Color(red: 94/255, green: 234/255, blue: 212/255).opacity(0.25)

    // MARK: - Text
    static let textPrimary   = Color.white.opacity(0.85)
    static let textSecondary = Color.white.opacity(0.4)
    static let textMuted     = Color.white.opacity(0.25)

    // MARK: - Borders
    static let border       = Color(red: 94/255, green: 234/255, blue: 212/255).opacity(0.15)
    static let borderHover  = Color(red: 94/255, green: 234/255, blue: 212/255).opacity(0.30)

    // MARK: - Shadows
    static let shadowPanel = Color.black.opacity(0.5)
    static let shadowRadius: CGFloat = 24

    // MARK: - Dimensions
    static let cornerRadius: CGFloat = 14
    static let borderWidth: CGFloat = 1
}

// MARK: - Color hex initializer

extension Color {
    /// Initialise from a 6-digit hex string (without leading `#`).
    /// Example: `Color(hex: "c4b5fd")`
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: .alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a: UInt64, r: UInt64, g: UInt64, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}
