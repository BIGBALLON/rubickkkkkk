import SwiftUI

/// Loading skeleton — 5 placeholder rows with a shimmer animation.
struct PulsarSkeletonView: View {

    @State private var shimmerPhase: CGFloat = -1.0

    var body: some View {
        VStack(spacing: 0) {
            ForEach(0..<5, id: \.self) { _ in
                skeletonRow
                Divider()
                    .background(Color.white.opacity(0.04))
            }
        }
        .onAppear {
            withAnimation(
                .linear(duration: 1.4)
                .repeatForever(autoreverses: false)
            ) {
                shimmerPhase = 1.0
            }
        }
    }

    private var skeletonRow: some View {
        HStack(spacing: 13) {
            // Thumbnail placeholder
            RoundedRectangle(cornerRadius: 10)
                .fill(shimmerFill)
                .frame(width: 44, height: 44)

            VStack(alignment: .leading, spacing: 6) {
                // Filename placeholder
                RoundedRectangle(cornerRadius: 4)
                    .fill(shimmerFill)
                    .frame(width: 160, height: 12)
                // Path placeholder
                RoundedRectangle(cornerRadius: 4)
                    .fill(shimmerFillDim)
                    .frame(width: 100, height: 10)
            }

            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 9)
    }

    /// Shimmer gradient fill.
    private var shimmerFill: some ShapeStyle {
        LinearGradient(
            stops: [
                .init(color: Color.white.opacity(0.05), location: max(0, shimmerPhase - 0.4)),
                .init(color: Color.white.opacity(0.12), location: shimmerPhase),
                .init(color: Color.white.opacity(0.05), location: min(1, shimmerPhase + 0.4)),
            ],
            startPoint: .leading,
            endPoint: .trailing
        )
    }

    private var shimmerFillDim: some ShapeStyle {
        LinearGradient(
            stops: [
                .init(color: Color.white.opacity(0.03), location: max(0, shimmerPhase - 0.4)),
                .init(color: Color.white.opacity(0.08), location: shimmerPhase),
                .init(color: Color.white.opacity(0.03), location: min(1, shimmerPhase + 0.4)),
            ],
            startPoint: .leading,
            endPoint: .trailing
        )
    }
}
