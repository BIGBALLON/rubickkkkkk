import SwiftUI

/// Pulsar bottom hint bar.
///
/// Shows keyboard shortcut hints (↑↓ navigate / ↵ open / ⌥Space×2 nebula)
/// and the result count on the right.
/// `resultCount`: currently displayed count (viewport), `totalCount`: backend total hits.
struct PulsarHintBar: View {

    let resultCount: Int
    let totalCount: Int

    var body: some View {
        HStack(spacing: 0) {
            hint(key: "↑↓", label: "navigate")
            hint(key: "↵",  label: "open")
            hint(key: "⌥Space×2", label: "nebula")

            Spacer()

            if resultCount > 0 {
                let label = totalCount > resultCount
                    ? "\(resultCount) of \(totalCount)"
                    : "\(resultCount) result\(resultCount == 1 ? "" : "s")"
                Text(label)
                    .font(.system(size: 10.5))
                    .foregroundStyle(DS.textMuted)
                    .tracking(0.03 * 10.5)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 9)
        .padding(.bottom, 10)
        .background(DS.bgDeep.opacity(0.5))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.white.opacity(0.05))
                .frame(height: 0.5)
        }
    }

    private func hint(key: String, label: String) -> some View {
        HStack(spacing: 5) {
            kbdChip(key)
            Text(label)
                .font(.system(size: 10.5))
                .foregroundStyle(Color.white.opacity(0.20))
        }
        .padding(.trailing, 16)
    }

    private func kbdChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10))
            .foregroundStyle(Color.white.opacity(0.35))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(Color.white.opacity(0.07))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.white.opacity(0.12), lineWidth: 0.5)
            )
            .cornerRadius(4)
    }
}
