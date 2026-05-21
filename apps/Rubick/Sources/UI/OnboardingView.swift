import AppKit
import SwiftUI

/// First-launch onboarding — simplified 3-step flow:
///   0 → Welcome
///   1 → Service Setup (model download + backend status)
///   2 → Add Folders → "Get started" dismisses
struct OnboardingView: View {
    let onComplete: () -> Void

    @State private var step: Int = 0
    private static let totalSteps = 3

    var body: some View {
        VStack(spacing: 0) {
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            footer
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
        }
        .frame(width: 540, height: 480)
        .interactiveDismissDisabled()
    }

    @ViewBuilder
    private var content: some View {
        switch step {
        case 0:
            welcomePage
        case 1:
            ServiceSetupPage()
        case 2:
            addFoldersPage
        default:
            welcomePage
        }
    }

    // MARK: - Step 0: Welcome

    private var welcomePage: some View {
        VStack(spacing: 18) {
            Spacer(minLength: 16)
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
                .symbolRenderingMode(.hierarchical)
            Text("Welcome to Rubick")
                .font(.title.weight(.semibold))
                .multilineTextAlignment(.center)
            Text("Grand Magus doctrine: recall, don't invent — pure MLX backend, zero uplink.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
                .lineSpacing(2)
            Spacer(minLength: 16)
        }
        .padding(.horizontal, 24)
    }

    // MARK: - Step 2: Add folders

    private var addFoldersPage: some View {
        VStack(spacing: 18) {
            Spacer(minLength: 16)
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
                .symbolRenderingMode(.hierarchical)
            Text("Add your files")
                .font(.title.weight(.semibold))
                .multilineTextAlignment(.center)
            Text("""
                Point Rubick at a folder to start indexing. Indexing \
                happens in the background and keeps up with file \
                changes automatically.

                Tip: ⌥+Space opens quick search from any app, \
                ⌥+Space×2 opens the Nebula 3D map.
                """)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
                .lineSpacing(2)
            Spacer(minLength: 16)
        }
        .padding(.horizontal, 24)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            Button("Skip") { onComplete() }
                .buttonStyle(.borderless)
                .controlSize(.small)

            Spacer()

            HStack(spacing: 6) {
                ForEach(0..<Self.totalSteps, id: \.self) { idx in
                    Circle()
                        .fill(idx == step ? Color.accentColor : Color.secondary.opacity(0.35))
                        .frame(width: 6, height: 6)
                }
            }

            Spacer()

            HStack(spacing: 8) {
                if step > 0 {
                    Button("Back") { step -= 1 }
                        .keyboardShortcut(.leftArrow, modifiers: [])
                }
                Button(step == Self.totalSteps - 1 ? "Get Started" : "Continue") {
                    if step < Self.totalSteps - 1 {
                        step += 1
                    } else {
                        onComplete()
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
    }
}

// MARK: - Step 1: Service Setup (model download + backend status)

/// Shows model download progress + backend readiness. User can pick
/// mirror source and trigger download. Progress bar shows live updates.
private struct ServiceSetupPage: View {
    @EnvironmentObject private var backend: BackendController
    @AppStorage("hf_endpoint") private var hfEndpoint: String = ""
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0
    @State private var downloadStatus: String = "idle"
    @State private var downloadError: String?
    @State private var models: [BackendModelInfo] = []

    private let mirrorOptions = [
        ("", "Auto (try official → mirror fallback)"),
        ("https://hf-mirror.com", "hf-mirror.com (recommended for China)"),
        ("https://huggingface.co", "huggingface.co (international)"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Service Setup")
                .font(.title2.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .center)

            // Backend status
            HStack(spacing: 10) {
                Circle()
                    .fill(backend.status.isReady ? Color.green : Color.orange)
                    .frame(width: 10, height: 10)
                Text("Backend: \(backend.status.label)")
                    .font(.callout)
                if !backend.status.isReady {
                    ProgressView().controlSize(.small)
                }
                Spacer()
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))

            // Model status
            if let model = models.first {
                HStack(spacing: 10) {
                    ModelStatusBadge(info: model)
                    Text(model.repo)
                        .font(.caption.monospaced())
                        .lineLimit(1)
                    Spacer()
                }
                .padding(12)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
            }

            // Mirror picker
            VStack(alignment: .leading, spacing: 6) {
                Text("Download source:")
                    .font(.subheadline.weight(.medium))
                Picker("", selection: $hfEndpoint) {
                    ForEach(mirrorOptions, id: \.0) { opt in
                        Text(opt.1).tag(opt.0)
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
            }

            // Download progress / trigger
            if isDownloading {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Downloading model…")
                            .font(.callout.weight(.medium))
                        Spacer()
                        Text("\(Int(downloadProgress * 100))%")
                            .font(.caption.monospacedDigit())
                    }
                    ProgressView(value: downloadProgress)
                        .progressViewStyle(.linear)
                    Text("~1.8 GB · This may take several minutes")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if downloadStatus == "complete" {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Model downloaded successfully!")
                        .font(.callout)
                }
            } else {
                Button {
                    Task { await triggerDownload() }
                } label: {
                    Label("Download Model (~1.8 GB)", systemImage: "arrow.down.circle")
                }
                .disabled(!backend.status.isReady)
            }

            if let error = downloadError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(24)
        .task { await refreshModels() }
    }

    private func triggerDownload() async {
        guard let client = backend.client else { return }
        isDownloading = true
        downloadError = nil
        downloadProgress = 0

        do {
            _ = try await client.triggerModelDownload(endpoint: hfEndpoint)
            while isDownloading {
                try await Task.sleep(nanoseconds: 1_500_000_000)
                let progress = try await client.modelDownloadProgress()
                let total = max(progress.totalBytes, 1)
                downloadProgress = Double(progress.downloadedBytes) / Double(total)
                downloadStatus = progress.status
                if progress.status == "complete" {
                    isDownloading = false
                    await refreshModels()
                } else if progress.status == "error" {
                    isDownloading = false
                    downloadError = progress.error ?? "Download failed"
                }
            }
        } catch {
            isDownloading = false
            downloadError = error.localizedDescription
        }
    }

    private func refreshModels() async {
        guard let client = backend.client else { return }
        do {
            let response = try await client.healthzModel()
            models = response.models
            // Auto-detect if already downloaded
            if let m = models.first, m.downloadStatus == .complete {
                downloadStatus = "complete"
            }
        } catch {
            // Ignore — backend may not be ready yet
        }
    }
}
