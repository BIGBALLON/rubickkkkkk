import SwiftUI

/// Pulsar search bar — top input area.
///
/// Uses a native TextField directly (no overlay trick) so that all
/// standard text editing features work: arrow keys, selection, Cmd+A,
/// Cmd+C, etc. The gradient first-word effect is sacrificed in favour
/// of correct editing UX — a worthwhile trade for a search launcher.
struct PulsarSearchBar: View {

    @Binding var query: String
    @FocusState.Binding var isFocused: Bool

    var body: some View {
        HStack(spacing: 12) {
            gradientSearchIcon

            TextField("Search…", text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(Color.white.opacity(0.85))
                .focused($isFocused)

            Spacer(minLength: 0)

            escBadge
        }
        .padding(.leading, 16)
        .padding(.trailing, 20)
        .padding(.vertical, 15)
        .background(Color.white.opacity(0.01))
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.white.opacity(0.05))
                .frame(height: 0.5)
        }
    }

    // MARK: - Subviews

    private var gradientSearchIcon: some View {
        Image(systemName: "magnifyingglass")
            .font(.system(size: 15, weight: .medium))
            .foregroundStyle(PulsarColors.searchIcon)
    }

    private var escBadge: some View {
        Text("esc")
            .font(.system(size: 10.5))
            .foregroundStyle(Color.white.opacity(0.20))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Color.white.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(Color.white.opacity(0.09), lineWidth: 0.5)
            )
    }
}
