import AppKit
import SwiftUI

/// Rubick's Settings window — bound to ``Cmd+,`` automatically by
/// SwiftUI's ``Settings`` scene. Six tabs (General / Sources / Privacy /
/// Index / Model / About):
///
/// - **General** — hotkey display, search top-K, backend status / port.
/// - **Sources** — re-uses ``WatchedFoldersSidebar`` so menu-bar-only
///   workflows can manage folders without bringing the main window
///   forward.
/// - **Index** — live ``GET /index/stats`` panel, pause/resume, clear data.
/// - **Privacy** — always-on exclusion deny-list.
/// - **Model** — driven by ``GET /healthz/model`` (single embedding model).
/// - **About** — version, links, license attribution.
struct SettingsView: View {
    @EnvironmentObject private var backend: BackendController

    var body: some View {
        TabView {
            ServiceTab()
                .environmentObject(backend)
                .tabItem { Label("Service", systemImage: "server.rack") }

            SourcesTab()
                .environmentObject(backend)
                .tabItem { Label("Sources", systemImage: "folder") }

            IndexTab()
                .environmentObject(backend)
                .tabItem { Label("Index", systemImage: "chart.bar.doc.horizontal") }

            GeneralTab()
                .environmentObject(backend)
                .tabItem { Label("General", systemImage: "gearshape") }

            AboutTab()
                .tabItem { Label("About", systemImage: "info.circle") }
        }
        .frame(minWidth: 620, minHeight: 560)
        .padding(20)
    }
}

// MARK: - General tab

/// Settings → General — read-only summary of the app's globally-scoped
/// state: hotkey, backend connection, and where on-disk data lives.
/// "Open at login" / startup-mode / idle-unload toggles (per spec)
/// land once we wire ``SMAppService``; today the panel makes the
/// existing implicit defaults explicit so users can see what's
/// happening without poking at logs.
private struct GeneralTab: View {
    @EnvironmentObject private var backend: BackendController
    /// Search results TopK count, default 20, stored in UserDefaults.
    @AppStorage("search_top_k") private var searchTopK: Int = 20

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("General")
                .font(.title3.weight(.semibold))

            section("Search results") {
                LabeledContent("Top K results") {
                    HStack(spacing: 8) {
                        Stepper(
                            value: $searchTopK,
                            in: 1...50,
                            step: 1
                        ) {
                            Text("\(searchTopK)")
                                .frame(minWidth: 32, alignment: .trailing)
                                .monospacedDigit()
                        }
                        Text("results per search")
                            .foregroundStyle(.secondary)
                            .font(.callout)
                    }
                }
                Text("Applies to both Pulsar quick search and Nebula main window. Restart is not required.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            section("Permissions") {
                PermissionsSection()
                    .environmentObject(backend)
            }

            section("Watch mode") {
                WatchModePicker(service: backend.watchService)
            }

            Spacer()
        }
        .padding()
        .onAppear {
            if searchTopK > 50 { searchTopK = 50 }
            if searchTopK < 1 { searchTopK = 20 }
        }
    }

    private var dataDirHint: String {
        if let env = ProcessInfo.processInfo.environment["RUBICK_DATA_DIR"], !env.isEmpty {
            return "\(env) (RUBICK_DATA_DIR)"
        }
        return "~/Library/Application Support/Rubick/"
    }

    private func keyCap(_ label: String) -> some View {
        Text(label)
            .font(.body.weight(.semibold).monospaced())
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    private func section<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            content()
        }
    }
}

// MARK: - Sources tab

/// Settings → Sources — exposes the same watched-folders surface
/// the main window's sidebar does, so users who keep the main
/// window hidden (menu-bar / hotkey-only workflow) can still add
/// and remove folders. The user-editable exclusion-rules editor
/// lives under the Privacy tab.
private struct SourcesTab: View {
    @EnvironmentObject private var backend: BackendController

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Sources")
                .font(.title3.weight(.semibold))
            Text(
                "Folders Rubick watches for new / changed files. Same list "
                + "as the main window's sidebar — edits here propagate live."
            )
            .font(.callout)
            .foregroundStyle(.secondary)

            // Reuse the sidebar verbatim so any future polish ("Reveal
            // in Finder" context menu, status pill, etc.) reaches both
            // surfaces in one edit.
            WatchedFoldersSidebar(
                service: backend.watchService,
                store: backend.watchService.folders,
                enabled: backend.status.isReady
            )
            .frame(minHeight: 220)
            .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))

            Text(
                "Exclusion rules — `.git`, `node_modules`, `__pycache__`, "
                + "system bundles, and similar are dropped automatically "
                + "by the ingest walker. Add your own fnmatch globs from "
                + "the Privacy tab."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding()
    }
}

// MARK: - Privacy tab

/// Settings → Privacy — two surfaces:
///
/// 1. **Always excluded** (read-only). Hard-coded defaults the
///    backend's ingest walker always honours: hidden dirs + a small
///    deny-list (``node_modules``, ``__pycache__``, …). We hard-code
///    the *categories* on the Swift side so the panel is useful
///    before the backend boots, but pull the live dir-name list
///    from ``GET /settings`` so a future backend addition shows up
///    without a Swift recompile.
/// 2. **Your rules** (v1.x #3 — editable). fnmatch globs applied to
///    dir + file basenames by the ingest walker, AND-ed with the
///    always-on list. Persisted in ``settings.json`` server-side;
///    every add / remove PATCHes through immediately so the next
///    ingest job picks them up. Cap + per-pattern length match the
///    backend so the UI surfaces "you've hit the cap" before the
///    round-trip silently trims.
private struct PrivacyTab: View {
    @EnvironmentObject private var backend: BackendController

    /// Categories of always-on exclusions we describe with a fixed
    /// "why" copy. The actual dir-name set comes from the backend
    /// (``defaultExclusionDirNames``); we fall back to this for the
    /// pre-backend-ready render.
    private static let alwaysExcludedCategories: [(pattern: String, why: String)] = [
        ("Hidden dirs (.git / .svn / .hg / .venv / .cache …)", "version control + scratch"),
        ("Build / dep dirs (node_modules / __pycache__ / build / dist / target …)", "compiled artefacts"),
        ("System bundles (*.app / *.framework)", "macOS application packages"),
        ("Disk images (*.dmg / *.iso)", "binary blobs"),
        ("System directories (~/Library / ~/Applications)", "unless explicitly added"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Privacy")
                    .font(.title3.weight(.semibold))

                Text(
                    "Rubick stores everything on your Mac. We never collect "
                    + "telemetry; the only network call is the first-time "
                    + "embedding-model download from HuggingFace."
                )
                .font(.callout)
                .foregroundStyle(.secondary)

                alwaysExcludedCard

                UserExclusionRulesSection()
                    .environmentObject(backend)

                Text(
                    "Mark-folder-as-private + per-folder encrypted "
                    + "indexes are tracked for a future release."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding()
        }
    }

    @ViewBuilder
    private var alwaysExcludedCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Always excluded from indexing")
                .font(.subheadline.weight(.semibold))
            ForEach(Self.alwaysExcludedCategories, id: \.pattern) { row in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(row.pattern)
                        .font(.callout.monospaced())
                        .frame(maxWidth: 320, alignment: .leading)
                    Text(row.why)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - User exclusion rules (v1.x #3)

/// Editable list of fnmatch globs that the ingest walker filters
/// dir + file basenames against. PATCHes ``/settings`` on every
/// add / remove so a runtime change takes effect on the very next
/// ingest job — no restart needed.
///
/// Two intentional simplicities for v1:
/// - Patterns match the basename only (no anchored / recursive
///   ``**`` semantics). The UI hint copy says this explicitly so a
///   user expecting full ``.gitignore`` doesn't get surprised.
/// - Empty trim → drop (matches the backend sanitiser). A double
///   click on the empty "+ Add rule" submission is a no-op.
private struct UserExclusionRulesSection: View {
    @EnvironmentObject private var backend: BackendController

    @State private var settings: BackendChunkingSettings?
    @State private var draftPattern: String = ""
    @State private var loadError: String?
    @State private var saveError: String?
    @State private var pendingSave: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Your rules")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if pendingSave {
                    ProgressView().controlSize(.mini)
                }
                if let count = settings?.exclusionPatterns.count {
                    let cap = settings?.exclusionPatternLimits.maxCount ?? 64
                    Text("\(count) / \(cap)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            Text(
                "Add fnmatch globs to exclude folders or files by name. "
                + "Examples: `secrets` blocks any folder literally named "
                + "secrets; `*.tmp` blocks any file ending in `.tmp`; "
                + "`backup-*` blocks anything starting with that prefix. "
                + "Patterns match the basename only — anchored / "
                + "recursive ``**`` rules are not supported in v1."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)

            if let settings {
                rulesList(settings)
                addRow(settings)
            } else if let loadError {
                Text(loadError)
                    .font(.caption)
                    .foregroundStyle(.red)
            } else {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Loading…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let saveError {
                Text(saveError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
        .task { await refresh() }
    }

    @ViewBuilder
    private func rulesList(_ s: BackendChunkingSettings) -> some View {
        if s.exclusionPatterns.isEmpty {
            Text("No rules yet.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.vertical, 2)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(s.exclusionPatterns, id: \.self) { pattern in
                    HStack {
                        Text(pattern)
                            .font(.callout.monospaced())
                        Spacer()
                        Button {
                            Task { await remove(pattern: pattern) }
                        } label: {
                            Image(systemName: "trash")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                        .help("Remove this rule")
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    @ViewBuilder
    private func addRow(_ s: BackendChunkingSettings) -> some View {
        let atCap = s.exclusionPatterns.count
            >= s.exclusionPatternLimits.maxCount
        HStack {
            TextField("Add pattern (e.g. *.log)", text: $draftPattern)
                .textFieldStyle(.roundedBorder)
                .font(.callout.monospaced())
                .disabled(atCap || pendingSave)
                .onSubmit {
                    Task { await addDraft() }
                }
            Button {
                Task { await addDraft() }
            } label: {
                Label("Add", systemImage: "plus.circle")
            }
            .disabled(
                atCap
                    || pendingSave
                    || draftPattern.trimmingCharacters(in: .whitespaces).isEmpty
                    || draftPattern.count > s.exclusionPatternLimits.maxLength
            )
            .help(
                atCap
                ? "Maximum \(s.exclusionPatternLimits.maxCount) rules — "
                  + "remove one to add a new pattern."
                : "Add this glob to your exclusion list. The next "
                  + "ingest job (live or re-scan) will respect it."
            )
        }
    }

    // MARK: - Backend round-trips

    private func refresh() async {
        guard let client = backend.client else {
            loadError = "Backend not connected yet."
            return
        }
        do {
            settings = try await client.getSettings()
            loadError = nil
        } catch {
            loadError = "Failed to load: \(error.localizedDescription)"
        }
    }

    /// Submit the current ``draftPattern``. Trims whitespace, dedups
    /// against the live list, and PATCHes only if the result differs.
    /// On success the field is cleared.
    private func addDraft() async {
        guard let s = settings else { return }
        let trimmed = draftPattern.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        guard trimmed.count <= s.exclusionPatternLimits.maxLength else {
            saveError = "Pattern is too long (max "
                + "\(s.exclusionPatternLimits.maxLength) chars)."
            return
        }
        if s.exclusionPatterns.contains(trimmed) {
            // Duplicate — just clear the field.
            draftPattern = ""
            return
        }
        var next = s.exclusionPatterns
        next.append(trimmed)
        await save(next)
        draftPattern = ""
    }

    private func remove(pattern: String) async {
        guard let s = settings else { return }
        let next = s.exclusionPatterns.filter { $0 != pattern }
        await save(next)
    }

    private func save(_ patterns: [String]) async {
        guard let client = backend.client else {
            saveError = "Backend not connected yet."
            return
        }
        saveError = nil
        pendingSave = true
        defer { pendingSave = false }
        do {
            let snap = try await client.updateSettings(
                exclusionPatterns: patterns
            )
            // Mirror the backend's actual-applied list back into the
            // local view state. Keeping the other metadata fields
            // unchanged so we don't lose the bounds / defaults
            // envelope on a list edit.
            if let current = settings {
                settings = BackendChunkingSettings(
                    targetTokens: snap.targetTokens,
                    hardMaxTokens: snap.hardMaxTokens,
                    exclusionPatterns: snap.exclusionPatterns,
                    defaults: current.defaults,
                    bounds: current.bounds,
                    defaultExclusionDirNames:
                        current.defaultExclusionDirNames,
                    exclusionPatternLimits:
                        current.exclusionPatternLimits
                )
            }
        } catch {
            saveError = "Save failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Service tab (backend status + model + download)

/// Settings → Service — backend health, model download status, mirror picker.
/// Replaces the old "Model" tab with a unified service management view.
private struct ServiceTab: View {
    @EnvironmentObject private var backend: BackendController

    @State private var models: [BackendModelInfo] = []
    @State private var loadError: String?
    @State private var lastRefreshedAt: Date?
    @State private var isRefreshing = false
    @State private var confirmingClear: BackendModelInfo?
    @State private var clearingId: String?
    @State private var lastClearOutcome: String?
    @State private var clearError: String?
    /// Mirror endpoint selection
    @AppStorage("hf_endpoint") private var hfEndpoint: String = ""
    @State private var isDownloading = false
    @State private var downloadError: String?

    private let mirrorOptions = [
        ("", "Auto (official → mirror fallback)"),
        ("https://hf-mirror.com", "hf-mirror.com (China)"),
        ("https://huggingface.co", "huggingface.co (International)"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Service")
                    .font(.title3.weight(.semibold))

                // Backend status
                backendStatusCard

                // Mirror source picker
                mirrorPicker

                // Download status (if downloading)
                if isDownloading {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text("Downloading model… please wait")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(12)
                    .background(.blue.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                }

                // Model cards
                if let error = loadError {
                    errorBanner(error)
                } else if models.isEmpty {
                    placeholder
                } else {
                    ForEach(models) { info in
                        modelCard(info)
                    }
                }

                if let lastClearOutcome {
                    Text(lastClearOutcome).font(.caption).foregroundStyle(.secondary)
                }
                if let clearError {
                    Text(clearError).font(.caption).foregroundStyle(.red)
                }
                if let downloadError {
                    Text(downloadError).font(.caption).foregroundStyle(.red)
                }
            }
            .padding()
        }
        .task { await refresh() }
        .alert(
            "Re-download \(confirmingClear?.repo ?? "this model")?",
            isPresented: clearAlertBinding,
            presenting: confirmingClear
        ) { info in
            Button("Re-download", role: .destructive) {
                Task { await clearCache(for: info) }
            }
            Button("Cancel", role: .cancel) {}
        } message: { info in
            Text("This will delete the cached model files. Quit and reopen Rubick to trigger a fresh download.")
        }
    }

    // MARK: - Backend status card

    @ViewBuilder
    private var backendStatusCard: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(backend.status.isReady ? Color.green : Color.orange)
                .frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 2) {
                Text("Backend: \(backend.status.label)")
                    .font(.callout.weight(.medium))
                Text(dataDirHint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if !backend.status.isReady {
                ProgressView().controlSize(.small)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var dataDirHint: String {
        if let env = ProcessInfo.processInfo.environment["RUBICK_DATA_DIR"], !env.isEmpty {
            return env
        }
        return "~/Library/Application Support/Rubick/"
    }

    // MARK: - Mirror picker

    @ViewBuilder
    private var mirrorPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Download source")
                .font(.subheadline.weight(.semibold))
            Picker("", selection: $hfEndpoint) {
                ForEach(mirrorOptions, id: \.0) { opt in
                    Text(opt.1).tag(opt.0)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()

            if !hfEndpoint.isEmpty {
                Text("Using: \(hfEndpoint)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }

            Button {
                Task { await triggerDownload() }
            } label: {
                Label("Download Model Now", systemImage: "arrow.down.circle")
            }
            .disabled(isDownloading || !backend.status.isReady)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Trigger download

    private func triggerDownload() async {
        guard let client = backend.client else { return }
        isDownloading = true
        downloadError = nil

        do {
            _ = try await client.triggerModelDownload(endpoint: hfEndpoint)
            // Auto-refresh model card every 3s to show "On disk" progress
            while isDownloading {
                try await Task.sleep(nanoseconds: 3_000_000_000)
                await refresh()
                let progress = try await client.modelDownloadProgress()
                if progress.status == "complete" {
                    isDownloading = false
                    await refresh()
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

    // MARK: - Model helpers (shared between body and old ModelTab)

    @ViewBuilder
    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Waiting for the local backend to come up…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(
                "Model state is fetched from the running ``rubick_backend`` "
                + "process. The Settings sheet may have opened before the "
                + "uvicorn boot poll finished — Refresh once it does."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            VStack(alignment: .leading, spacing: 4) {
                Text("Could not load model status")
                    .font(.subheadline.weight(.semibold))
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Spacer()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.yellow.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func modelCard(_ info: BackendModelInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(Self.title(for: info.id))
                    .font(.subheadline.weight(.semibold))
                Spacer()
                ModelStatusBadge(info: info)
            }
            Text(info.repo)
                .font(.callout.monospaced())
                .textSelection(.enabled)
            Text(info.purpose)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 18) {
                LabeledContent("Declared", value: Self.formatBytes(info.declaredBytes))
                LabeledContent(
                    "On disk",
                    value: info.cacheBytes > 0 ? Self.formatBytes(info.cacheBytes) : "—"
                )
                Spacer()
                Button {
                    confirmingClear = info
                } label: {
                    if clearingId == info.id {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Clearing…")
                        }
                    } else {
                        Label("Re-download…", systemImage: "arrow.down.circle")
                    }
                }
                .disabled(!isReDownloadEnabled(for: info))
                .help(reDownloadHelp(for: info))
            }
            .font(.callout)
            if let path = info.cachePath {
                Text(path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    /// True when the user can meaningfully click "Re-download" right
    /// now: backend is up, no other clear is in flight on a different
    /// card, and the cache has something to clear (or the user wants
    /// to "ensure clean" against a partial download — we allow that).
    /// Disabled outright on an ``.absent`` cache so we don't seem to
    /// promise a download when all the operation would do is verify
    /// emptiness.
    private func isReDownloadEnabled(for info: BackendModelInfo) -> Bool {
        guard backend.status.isReady else { return false }
        if let clearingId, clearingId != info.id { return false }
        if isRefreshing { return false }
        return info.downloadStatus != .absent
    }

    private func reDownloadHelp(for info: BackendModelInfo) -> String {
        switch info.downloadStatus {
        case .absent:
            return "Nothing on disk to clear — relaunching Rubick will "
                + "download a fresh copy automatically."
        case .partial, .unknown:
            return "Delete the partial download and start over."
        case .complete:
            return "Delete the on-disk cache. The current session keeps "
                + "working until you quit; relaunching Rubick will then "
                + "re-download the model (~1.8 GB)."
        }
    }

    private var clearAlertBinding: Binding<Bool> {
        Binding(
            get: { confirmingClear != nil },
            set: { if !$0 { confirmingClear = nil } }
        )
    }

    private func clearAlertMessage(for info: BackendModelInfo) -> String {
        let onDisk = info.cacheBytes > 0
            ? Self.formatBytes(info.cacheBytes)
            : "—"
        return [
            "This will delete \(onDisk) of cached model files under "
                + "\(info.cachePath ?? "~/.cache/huggingface/hub/").",
            "",
            "Your current Rubick session keeps working — the model in "
                + "memory is untouched. The next time you quit and "
                + "reopen Rubick, fresh weights are downloaded from "
                + "HuggingFace (~1.8 GB, multi-minute on a typical link).",
            "",
            "Cancel if you didn't mean to.",
        ].joined(separator: "\n")
    }

    /// Run the backend delete + refresh the model card. Owns the
    /// optimistic local state (``clearingId`` spinner, outcome banner)
    /// and clears the alert before kicking off the work so a slow
    /// network doesn't leave a "you can't dismiss me" dialog parked
    /// on screen.
    private func clearCache(for info: BackendModelInfo) async {
        guard let client = backend.client else {
            clearError = "Backend not connected yet."
            return
        }
        confirmingClear = nil
        clearError = nil
        lastClearOutcome = nil
        clearingId = info.id
        defer { clearingId = nil }
        do {
            let result = try await client.clearModelCache(id: info.id)
            if result.wasPresent {
                lastClearOutcome =
                    "Cleared \(Self.formatBytes(result.deletedBytes)) "
                    + "for \(info.repo). Quit and reopen Rubick to "
                    + "trigger the re-download."
            } else {
                lastClearOutcome = "No cache to clear for \(info.repo) "
                    + "— next launch will fetch it fresh."
            }
        } catch {
            clearError = "Re-download failed: \(error.localizedDescription)"
        }
        await refresh()
    }

    // ``statusBadge`` lives in ``ModelStatusBadge.swift`` now —
    // the same view also renders inside the Onboarding "Model
    // setup" step, so centralizing the dot-color / label mapping
    // keeps the two call sites in lockstep.

    // MARK: - Async fetch

    private func refresh() async {
        guard let client = backend.client else {
            loadError = "Backend isn't connected yet (status: \(backend.status.label))."
            return
        }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            let response = try await client.healthzModel()
            models = response.models
            loadError = nil
            lastRefreshedAt = Date()
        } catch {
            loadError = error.localizedDescription
        }
    }

    // MARK: - Static helpers (no per-instance state)

    /// Map a backend ``id`` (currently always ``"embedding"``) to
    /// the human-readable card title shown above the repo line. Title
    /// is pure UI copy — keeping it Swift-side avoids a backend
    /// recompile every time we tweak the wording. The switch is
    /// future-proofed so a re-introduced second model lands without
    /// editing both ends.
    private static func title(for backendId: String) -> String {
        switch backendId {
        case "embedding": return "Embedding model (multimodal)"
        default:          return backendId.capitalized
        }
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useMB, .useGB]
        f.countStyle = .file
        return f.string(fromByteCount: bytes)
    }
}

// MARK: - Index tab

/// Settings → Index — aggregate counts (per modality + total docs)
/// pulled from ``GET /index/stats``. Refresh button on the title bar
/// re-fetches; the panel also auto-refreshes on first display so the
/// user never sees stale data on opening Settings.
///
/// "Clear All Index Data" calls ``DELETE /index/all``. "Pause / resume" is wired to
/// ``POST /index/{pause,resume}`` — toggling the button parks the
/// worker on its dispatch gate; in-flight ingest finishes but new
/// jobs accumulate in the queue until the user resumes.
private struct IndexTab: View {
    @EnvironmentObject private var backend: BackendController
    @State private var stats: IndexStats?
    @State private var loadError: String?
    @State private var isLoading = false
    @State private var isTogglingPause = false
    @State private var pauseError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Index")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    if isLoading {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Button {
                        Task { await refresh() }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .disabled(!backend.status.isReady || isLoading)
                    .help("Re-query the backend for current index counts.")
                }

                if let stats {
                    summaryCard(stats: stats)
                    modalityBreakdown(stats: stats)
                    rejectedRow(stats: stats)
                } else if let loadError {
                    Text(loadError)
                        .font(.callout)
                        .foregroundStyle(.red)
                } else {
                    Text(backend.status.isReady ? "Loading…" : backend.status.label)
                        .foregroundStyle(.secondary)
                }

                Divider()

                TextChunkingSection()
                    .environmentObject(backend)

                controlsFooter
            }
            .padding()
        }
        .task { await refresh() }
    }

    @ViewBuilder
    private func summaryCard(stats: IndexStats) -> some View {
        HStack(spacing: 24) {
            statBlock(
                label: "Documents",
                value: "\(stats.totalDocs)",
                systemImage: "doc.on.doc"
            )
            statBlock(
                label: "Chunks",
                value: "\(stats.totalChunks)",
                systemImage: "rectangle.split.3x1"
            )
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private func statBlock(label: String, value: String, systemImage: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: systemImage)
                .font(.title3)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.title2.weight(.semibold).monospacedDigit())
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func modalityBreakdown(stats: IndexStats) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("By modality")
                .font(.subheadline.weight(.semibold))
            // Render zero-count rows in a muted style so an empty
            // modality is visible (helpful for "have I indexed any
            // video yet?") without dominating the panel. The
            // ``rejected`` bucket is handled separately by
            // ``rejectedRow`` so successes and failures don't share
            // visual real estate.
            ForEach(
                stats.byModalitySorted.filter { $0.modality != "rejected" },
                id: \.modality
            ) { row in
                HStack {
                    let modality = Modality(rawString: row.modality)
                    Label(modality.displayLabel.capitalized, systemImage: modality.symbolName)
                        .foregroundStyle(row.count > 0 ? .primary : .secondary)
                    Spacer()
                    Text("\(row.count)")
                        .font(.body.monospacedDigit())
                        .foregroundStyle(row.count > 0 ? .primary : .secondary)
                }
                .padding(.vertical, 1)
            }
        }
    }

    /// Files that failed to ingest (oversize, decode-failed, or
    /// otherwise rejected by a pipeline). Lives in its own visual row
    /// — mixing it in with the success modalities made it look like
    /// "0 video + 3 image + 4 rejected" was 7 things you'd want to
    /// search, which it isn't. Suppressed entirely when nothing was
    /// rejected.
    @ViewBuilder
    private func rejectedRow(stats: IndexStats) -> some View {
        let rejectedChunks = stats.byModality["rejected"] ?? 0
        if rejectedChunks > 0 {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Skipped — \(rejectedChunks) file\(rejectedChunks == 1 ? "" : "s")")
                        .font(.callout.weight(.medium))
                    Text(
                        "Files Rubick couldn't ingest (oversize, decode-failed, "
                        + "unsupported format)."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    @State private var isClearingAll = false
    @State private var clearAllOutcome: String?
    @State private var confirmClearAll = false

    @ViewBuilder
    private var controlsFooter: some View {
        Divider()
        VStack(alignment: .leading, spacing: 10) {
            pauseRow
            HStack {
                Button(role: .destructive) {
                    confirmClearAll = true
                } label: {
                    if isClearingAll {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Clearing…")
                        }
                    } else {
                        Label("Clear All Index Data", systemImage: "trash")
                    }
                }
                .disabled(!backend.status.isReady || isClearingAll)
                Spacer()
            }
            if let clearAllOutcome {
                Text(clearAllOutcome)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .confirmationDialog(
            "Clear all indexed data?",
            isPresented: $confirmClearAll,
            titleVisibility: .visible
        ) {
            Button("Clear Everything", role: .destructive) {
                Task { await clearAllIndex() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes all indexed chunks, thumbnails, and the Nebula map. Your source files are untouched. You'll need to re-index after this.")
        }
    }

    private func clearAllIndex() async {
        guard let client = backend.client else { return }
        isClearingAll = true
        defer { isClearingAll = false }
        do {
            let res = try await client.clearAllIndex()
            clearAllOutcome = "Cleared \(res.deletedDocs) document(s). Re-index when ready."
        } catch {
            clearAllOutcome = "Error: \(error.localizedDescription)"
        }
        await refresh()
    }

    /// Pause / Resume control + live "is paused?" status line.
    ///
    /// Optimistic-ish: the button delegates to
    /// ``BackendController.setIndexPaused``, which performs the POST
    /// and re-publishes ``indexQueueStatus`` on success. On failure
    /// we surface a one-liner so the user sees something more
    /// actionable than a silent no-op (the next refresh edge will
    /// re-sync the truth either way).
    ///
    /// The button label tracks ``backend.indexQueueStatus?.paused``
    /// rather than a local state mirror so other surfaces (e.g. a
    /// future menu-bar quick-toggle) can flip the gate without us
    /// going out of sync.
    @ViewBuilder
    private var pauseRow: some View {
        let qs = backend.indexQueueStatus
        let isPaused = qs?.paused ?? false
        HStack(spacing: 10) {
            Button {
                Task { await togglePause(to: !isPaused) }
            } label: {
                if isTogglingPause {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.mini)
                        Text(isPaused ? "Resuming…" : "Pausing…")
                    }
                } else {
                    Label(
                        isPaused ? "Resume indexing" : "Pause indexing",
                        systemImage: isPaused ? "play.fill" : "pause.fill"
                    )
                }
            }
            .disabled(
                !backend.status.isReady || isTogglingPause || qs == nil
            )
            .help(
                isPaused
                ? "Let the backend drain any jobs that piled up while paused."
                : "Stop the backend from picking up new ingest jobs. "
                  + "In-flight work finishes; new jobs queue until you resume."
            )

            statusLabel(isPaused: isPaused, pending: qs?.pending)
            Spacer()
        }
        if let pauseError {
            Text(pauseError)
                .font(.caption)
                .foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private func statusLabel(isPaused: Bool, pending: Int?) -> some View {
        if isPaused {
            HStack(spacing: 4) {
                Image(systemName: "pause.circle.fill")
                    .foregroundStyle(.orange)
                Text(pauseSummary(pending: pending ?? 0))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if let pending, pending > 0 {
            Text("\(pending) job\(pending == 1 ? "" : "s") in flight")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else if backend.status.isReady {
            Text("Idle — indexing as files change.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func pauseSummary(pending: Int) -> String {
        if pending == 0 {
            return "Paused — nothing queued."
        }
        return "Paused — \(pending) job\(pending == 1 ? "" : "s") waiting."
    }

    private func togglePause(to nextPaused: Bool) async {
        pauseError = nil
        isTogglingPause = true
        defer { isTogglingPause = false }
        do {
            try await backend.setIndexPaused(nextPaused)
        } catch {
            pauseError = "Couldn't \(nextPaused ? "pause" : "resume")"
                + " indexing: \(error.localizedDescription)"
        }
    }

    /// Re-fetch ``GET /index/stats`` for the summary panel.
    private func refresh() async {
        guard let client = backend.client else {
            loadError = "Backend not connected yet."
            return
        }
        isLoading = true
        defer { isLoading = false }
        do {
            stats = try await client.indexStats()
            loadError = nil
        } catch {
            loadError = "Failed to load stats: \(error.localizedDescription)"
        }
    }
}

// MARK: - About tab

/// Settings → About — version + links.
private struct AboutTab: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Rubick")
                    .font(.largeTitle.weight(.semibold))
                Text(Self.appVersionString())
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Text("Local multimodal embedding-powered search for macOS.")
                .foregroundStyle(.secondary)
            HStack(spacing: 14) {
                Link("GitHub", destination: URL(string: "https://github.com/BIGBALLON/rubickkkkkk")!)
                Link("Changelog", destination: URL(string: "https://github.com/BIGBALLON/rubickkkkkk/blob/main/CHANGELOG.md")!)
            }
            .font(.callout)
            Spacer()
            Text("Code: MIT · Model: CC BY-NC 4.0 (non-commercial)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private static func appVersionString() -> String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "v\(short) (\(build))"
    }
}

// MARK: - Permissions section (v1.x #2)

/// Settings → General "Permissions" — compact mirror of the
/// Onboarding step 5 card. Re-fetches ``GET /healthz/permissions``
/// on appearance + on user-driven re-check; the same surface a
/// future "Accessibility" or "Screen Recording" probe would extend
/// (just add another row).
///
/// Hidden when the probe reports a non-Darwin platform — there's no
/// macOS permission concept on Linux dev VMs so any UI here would
/// be misleading.
private struct PermissionsSection: View {
    @EnvironmentObject private var backend: BackendController

    @State private var state: BackendPermissions?
    @State private var loadError: String?
    @State private var isRefreshing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let fda = state?.fullDiskAccess, fda.isApplicable {
                fdaRow(fda)
            } else if let loadError {
                Text(loadError)
                    .font(.caption)
                    .foregroundStyle(.red)
            } else if state == nil {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Checking permissions…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("Not applicable on this platform.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .task { await refresh() }
    }

    @ViewBuilder
    private func fdaRow(_ fda: FullDiskAccessState) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(fda.granted ? Color.green : Color.secondary)
                .frame(width: 8, height: 8)
                .padding(.top, 6)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text("Full Disk Access")
                        .font(.callout.weight(.medium))
                    Text(fda.granted ? "Granted" : "Not granted")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                }
                Text(detailText(fda))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                if !fda.granted {
                    Button {
                        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles")!)
                    } label: {
                        Label(
                            "Open Settings…",
                            systemImage: "arrow.up.right.square"
                        )
                    }
                    .controlSize(.small)
                }
                if isRefreshing {
                    ProgressView().controlSize(.mini)
                } else {
                    Button {
                        Task { await refresh() }
                    } label: {
                        Label("Re-check", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            }
        }
    }

    private func detailText(_ fda: FullDiskAccessState) -> String {
        if fda.granted {
            return "Rubick can read system-protected directories."
        }
        return "Optional. Grant it to watch ~/Library or other "
            + "protected roots; the standard Documents / Desktop / "
            + "Downloads flow already works without it."
    }

    private func refresh() async {
        guard let client = backend.client else {
            loadError = nil
            return
        }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            state = try await client.healthzPermissions()
            loadError = nil
        } catch {
            loadError = "Couldn't reach the local backend: "
                + error.localizedDescription
        }
    }
}

// MARK: - Watch-mode picker

/// Settings → General → "Watch mode" section. Three radio rows
/// (live / scheduled / manual) plus an interval stepper that
/// appears only when ``scheduled`` is selected.
///
/// Backed by ``WatchService.watchMode`` (``@Published``,
/// auto-persisted to ``UserDefaults``). Flipping a row writes back
/// immediately — no separate "Save" button — matching the rest of
/// the Settings panel's instant-apply convention.
private struct WatchModePicker: View {
    @ObservedObject var service: WatchService

    /// Local mirror of the scheduled-mode interval so the stepper
    /// can drive a slider-style int without us round-tripping the
    /// ``WatchMode`` enum on every increment. Synced with
    /// ``service.watchMode`` on appear and on each pick.
    @State private var scheduledMinutes: Int = WatchMode.defaultScheduledIntervalMinutes

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            modeRow(
                isOn: isScheduled,
                label: "Scheduled rescan",
                detail: "Re-scan every N minutes (default). Editing files in a "
                    + "watched folder doesn't trigger an immediate index — it "
                    + "lands on the next scheduled pass. Lower memory pressure "
                    + "for large folders.",
                onSelect: { service.watchMode = .scheduled(intervalMinutes: scheduledMinutes) }
            )
            if isScheduled {
                HStack {
                    Spacer().frame(width: 24)
                    Stepper(
                        value: Binding(
                            get: { scheduledMinutes },
                            set: { newValue in
                                scheduledMinutes = newValue
                                service.watchMode = .scheduled(intervalMinutes: newValue)
                            }
                        ),
                        in: WatchMode.scheduledIntervalRange,
                        step: 5
                    ) {
                        Text("Every \(scheduledMinutes) min")
                            .font(.callout.monospacedDigit())
                    }
                    .controlSize(.small)
                    Spacer()
                }
                .padding(.top, 2)
            }
            modeRow(
                isOn: isLive,
                label: "Real-time",
                detail: "Index every change as files are saved (FSEvents). "
                    + "Best for actively-edited folders; floods the queue on "
                    + "multi-thousand-image folders.",
                onSelect: { service.watchMode = .live }
            )
            modeRow(
                isOn: isManual,
                label: "Manual only",
                detail: "Track changes silently; you choose when to re-index "
                    + "from the sidebar's refresh button.",
                onSelect: { service.watchMode = .manual }
            )
        }
        .onAppear { syncScheduledMinutes() }
        .onChange(of: service.watchMode) { _ in syncScheduledMinutes() }
    }

    @ViewBuilder
    private func modeRow(
        isOn: Bool,
        label: String,
        detail: String,
        onSelect: @escaping () -> Void
    ) -> some View {
        Button(action: onSelect) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: isOn ? "largecircle.fill.circle" : "circle")
                    .font(.callout)
                    .foregroundStyle(isOn ? Color.accentColor : .secondary)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.callout.weight(.medium))
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
            }
            .contentShape(Rectangle())
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? [.isSelected] : [])
    }

    private var isLive: Bool {
        if case .live = service.watchMode { return true }
        return false
    }

    private var isScheduled: Bool {
        if case .scheduled = service.watchMode { return true }
        return false
    }

    private var isManual: Bool {
        if case .manual = service.watchMode { return true }
        return false
    }

    /// Pull the scheduled interval out of the live mode so the
    /// stepper UI stays in sync if the mode is mutated elsewhere
    /// (Sources tab, future MenuBar quick-toggle, etc.). No-op
    /// when the active mode isn't ``.scheduled``.
    private func syncScheduledMinutes() {
        if case .scheduled(let m) = service.watchMode {
            scheduledMinutes = m
        }
    }
}

// MARK: - Text-chunking settings

/// Settings → Index → "Text chunking" section. Three preset radio
/// rows (Small / Standard / Large) plus a Custom mode with two
/// inline steppers. PATCHes ``/settings`` on every change; persists
/// server-side to ``RUBICK_DATA_DIR/settings.json``.
///
/// Existing chunks were generated with whatever parameters were
/// active at the time, so changing the knobs only affects *future*
/// ingest. We surface that asymmetry with a yellow advisory banner
/// + a "Re-index everything" button that walks the watched-folders
/// list, clears each via ``DELETE /index/by-path-prefix``, then
/// re-kicks. Cheap on backend dedup since most files won't change
/// bytes between deletion and re-ingest.
private struct TextChunkingSection: View {
    @EnvironmentObject private var backend: BackendController

    @State private var settings: BackendChunkingSettings?
    @State private var loadError: String?
    /// Most recent successfully-applied (target, hardMax) — used
    /// to detect "user changed knobs since the last ingest" so the
    /// reindex banner can light up.
    @State private var initialSnapshot: BackendChunkingSnapshot?
    @State private var customTarget: Int = 0
    @State private var customHardMax: Int = 0
    @State private var isReindexing: Bool = false
    @State private var reindexOutcome: String?
    @State private var reindexError: String?

    /// Static chunk-size presets (match Settings → Index tunables).
    /// Keeping them const + module-local (rather than fetched from the
    /// backend) means a clean install renders the cards even before
    /// the backend boots.
    private struct Preset: Identifiable, Equatable {
        let id: String
        let label: String
        let target: Int
        let hardMax: Int
        let blurb: String
    }

    private static let presets: [Preset] = [
        Preset(
            id: "standard",
            label: "Standard",
            target: 2048,
            hardMax: 6144,
            blurb: "Recommended — balanced precision and context (default)."
        ),
        Preset(
            id: "large",
            label: "Large",
            target: 4096,
            hardMax: 8192,
            blurb: "Best for long technical docs / academic papers."
        ),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Text chunking")
                .font(.subheadline.weight(.semibold))
            Text(
                "Controls how Markdown / plain-text files are split before "
                + "embedding. Larger chunks reduce semantic breaks; smaller "
                + "chunks give finer-grained search hits."
            )
            .font(.caption)
            .foregroundStyle(.secondary)

            if let s = settings {
                presetRows(current: s)
                if isCustomActive(current: s) {
                    customStepperRow(bounds: s.bounds)
                }
                if shouldShowReindexBanner(current: s) {
                    reindexBanner
                }
            } else if let err = loadError {
                Text(err).font(.caption).foregroundStyle(.red)
            } else {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Loading…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .task { await refresh() }
    }

    // MARK: - Preset rows

    @ViewBuilder
    private func presetRows(current: BackendChunkingSettings) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Self.presets) { preset in
                presetRow(
                    preset: preset,
                    isOn: matches(preset: preset, current: current)
                )
            }
            customRow(isOn: isCustomActive(current: current))
        }
    }

    @ViewBuilder
    private func presetRow(preset: Preset, isOn: Bool) -> some View {
        Button {
            Task { await apply(target: preset.target, hardMax: preset.hardMax) }
        } label: {
            HStack(alignment: .top, spacing: 10) {
                radio(isOn: isOn)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(preset.label)
                            .font(.callout.weight(.medium))
                        Text("target \(preset.target) · max \(preset.hardMax)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Text(preset.blurb)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .contentShape(Rectangle())
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? [.isSelected] : [])
    }

    @ViewBuilder
    private func customRow(isOn: Bool) -> some View {
        Button {
            // Selecting Custom keeps the current values — no PATCH
            // is sent; the user is expected to drive the steppers
            // next. Each stepper edit independently PATCHes through.
        } label: {
            HStack(alignment: .top, spacing: 10) {
                radio(isOn: isOn)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Custom")
                        .font(.callout.weight(.medium))
                    Text("Adjust target + hard-max independently below.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .contentShape(Rectangle())
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .disabled(true) // selection is implicit when no preset matches
        .opacity(isOn ? 1.0 : 0.7)
        .accessibilityAddTraits(isOn ? [.isSelected] : [])
    }

    @ViewBuilder
    private func radio(isOn: Bool) -> some View {
        Image(systemName: isOn ? "largecircle.fill.circle" : "circle")
            .font(.callout)
            .foregroundStyle(isOn ? Color.accentColor : .secondary)
            .frame(width: 18)
    }

    // MARK: - Custom steppers

    @ViewBuilder
    private func customStepperRow(bounds: BackendChunkingSettings.BoundsPair) -> some View {
        HStack {
            Spacer().frame(width: 26)
            Stepper(
                value: Binding(
                    get: { customTarget },
                    set: { newValue in
                        let clamped = newValue.clamped(to: bounds.targetTokens)
                        customTarget = clamped
                        Task {
                            await apply(target: clamped, hardMax: customHardMax)
                        }
                    }
                ),
                in: bounds.targetTokens,
                step: 100
            ) {
                Text("Target \(customTarget) tokens")
                    .font(.callout.monospacedDigit())
            }
            .controlSize(.small)
            Stepper(
                value: Binding(
                    get: { customHardMax },
                    set: { newValue in
                        let clamped = newValue.clamped(to: bounds.hardMaxTokens)
                        customHardMax = clamped
                        Task {
                            await apply(target: customTarget, hardMax: clamped)
                        }
                    }
                ),
                in: bounds.hardMaxTokens,
                step: 100
            ) {
                Text("Max \(customHardMax) tokens")
                    .font(.callout.monospacedDigit())
            }
            .controlSize(.small)
            Spacer()
        }
        .padding(.top, 2)
    }

    // MARK: - Reindex banner

    @ViewBuilder
    private var reindexBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Existing chunks use the previous settings.")
                        .font(.caption.weight(.medium))
                    Text(
                        "They'll keep working, but new chunks (from edits or "
                        + "newly-watched folders) will use the new values. "
                        + "Re-index to apply the new settings everywhere."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
            HStack(spacing: 8) {
                Button {
                    Task { await reindexEverything() }
                } label: {
                    if isReindexing {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Re-indexing…")
                        }
                    } else {
                        Text("Re-index everything")
                    }
                }
                .disabled(isReindexing || !backend.status.isReady)
                Spacer()
                if let outcome = reindexOutcome {
                    Text(outcome)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let err = reindexError {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
        .padding(10)
        .background(.yellow.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
        .padding(.top, 6)
    }

    // MARK: - State derivation

    private func matches(
        preset: Preset,
        current: BackendChunkingSettings
    ) -> Bool {
        current.targetTokens == preset.target
            && current.hardMaxTokens == preset.hardMax
    }

    private func isCustomActive(current: BackendChunkingSettings) -> Bool {
        !Self.presets.contains { matches(preset: $0, current: current) }
    }

    private func shouldShowReindexBanner(current: BackendChunkingSettings) -> Bool {
        guard let initial = initialSnapshot else { return false }
        return initial.targetTokens != current.targetTokens
            || initial.hardMaxTokens != current.hardMaxTokens
    }

    // MARK: - Backend round-trips

    private func refresh() async {
        guard let client = backend.client else {
            loadError = "Backend not connected yet."
            return
        }
        do {
            let s = try await client.getSettings()
            settings = s
            customTarget = s.targetTokens
            customHardMax = s.hardMaxTokens
            // Lock in the "value at section appearance" so a later
            // change can compare against it for the reindex banner.
            // ``initialSnapshot`` survives later refresh()es so the
            // banner keeps showing until the user either reindexes
            // or reverts the values manually.
            if initialSnapshot == nil {
                initialSnapshot = BackendChunkingSnapshot(
                    targetTokens: s.targetTokens,
                    hardMaxTokens: s.hardMaxTokens
                )
            }
            loadError = nil
        } catch {
            loadError = "Failed to load settings: \(error.localizedDescription)"
        }
    }

    private func apply(target: Int, hardMax: Int) async {
        guard let client = backend.client, var s = settings else { return }
        do {
            let snap = try await client.updateSettings(
                targetTokens: target,
                hardMaxTokens: hardMax
            )
            // Mirror the backend's clamped values back into the
            // local view state so the radio rows + steppers reflect
            // *what was applied* rather than *what was requested*.
            s = BackendChunkingSettings(
                targetTokens: snap.targetTokens,
                hardMaxTokens: snap.hardMaxTokens,
                exclusionPatterns: snap.exclusionPatterns,
                defaults: s.defaults,
                bounds: s.bounds,
                defaultExclusionDirNames: s.defaultExclusionDirNames,
                exclusionPatternLimits: s.exclusionPatternLimits
            )
            settings = s
            customTarget = snap.targetTokens
            customHardMax = snap.hardMaxTokens
        } catch {
            loadError = "Failed to save settings: \(error.localizedDescription)"
        }
    }

    /// Re-index every watched folder under the new chunking
    /// parameters. Two phases:
    ///
    /// 1. ``DELETE /index/by-path-prefix`` per folder — removes the
    ///    old chunks and their thumbnails.
    /// 2. ``WatchService.refreshAll()`` — re-kicks the standard
    ///    ingest path on each folder, which now uses the new
    ///    parameters.
    ///
    /// Composes existing primitives (``DELETE /index/by-path-prefix`` per
    /// folder + ``refreshAll``) instead of adding
    /// a new backend "force-reingest" endpoint. The flow is async
    /// + sequential so a hot search isn't competing with N
    /// parallel index jobs.
    private func reindexEverything() async {
        guard let client = backend.client else {
            reindexError = "Backend not ready."
            return
        }
        let watchedFolders = backend.watchService.folders.folders
        guard !watchedFolders.isEmpty else {
            reindexOutcome = "No folders to re-index."
            return
        }
        reindexError = nil
        reindexOutcome = nil
        isReindexing = true
        defer { isReindexing = false }

        var deletedDocs = 0
        for folder in watchedFolders {
            do {
                let res = try await client.clearByPathPrefix(
                    folder.url.standardized.path
                )
                deletedDocs += res.deletedDocs
            } catch {
                reindexError = "Clear failed for \(folder.displayName): "
                    + "\(error.localizedDescription)"
                return
            }
        }
        await backend.watchService.refreshAll()
        // Snapshot the freshly-applied values so the banner stops
        // showing once the work is done. Reindex outcome shows the
        // headline number; per-folder errors (if any) end up in
        // each folder's ``WatchedFolder.status`` row already.
        if let s = settings {
            initialSnapshot = BackendChunkingSnapshot(
                targetTokens: s.targetTokens,
                hardMaxTokens: s.hardMaxTokens
            )
        }
        reindexOutcome = "Re-indexed \(watchedFolders.count) folder(s); "
            + "cleared \(deletedDocs) prior doc(s)."
    }
}

// MARK: - Numeric helpers

private extension Comparable {
    /// SwiftUI ``ClampedTo`` — Foundation provides ``clamp(_:to:)``
    /// in newer SDKs but we keep our own one-liner so the call site
    /// reads naturally on every macOS we still support.
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
