import SwiftUI

/// Pulsar-specific tokens derived from the shared Deep Space palette.
enum PulsarColors {
    // Panel
    static let panelFill = DS.bgDeep.opacity(0.95)

    // Border: single-color violet (no more multi-color gradient)
    static let panelBorder = DS.accentGlow

    // Selected row
    static let selectedRowFill = DS.accentSubtle

    // Score pill
    static let scorePillFill = DS.accent.opacity(0.2)

    // Search icon
    static let searchIcon = DS.accent.opacity(0.85)

    // Cursor / caret
    static let cursor = DS.accent
}
