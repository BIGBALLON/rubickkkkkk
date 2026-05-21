import ProjectDescription

/// Rubick — local multimodal embedding-powered search for macOS.
///
/// Single SwiftUI app target (`Rubick`), macOS 13+ / Apple Silicon only.
/// Not sandboxed — relies on Notarization instead (local folder access +
/// Python subprocess).
let project = Project(
    name: "Rubick",
    organizationName: "BIGBALLON",
    options: .options(
        defaultKnownRegions: ["en", "Base"],
        developmentRegion: "en"
    ),
    settings: .settings(
        base: [
            // Apple Silicon only — MLX and our embedded Python all
            // require arm64.
            "ARCHS": "arm64",
            "ONLY_ACTIVE_ARCH": "YES",
            "SWIFT_VERSION": "5.10",
            "MACOSX_DEPLOYMENT_TARGET": "13.0",
            // Modern concurrency checks — we want strict warnings now,
            // not Swift 6 surprises later.
            "SWIFT_STRICT_CONCURRENCY": "complete",
            // No team / no provisioning for local dev builds; release
            // builds will inject these via env vars / CI.
            "CODE_SIGN_STYLE": "Automatic",
            "CODE_SIGN_IDENTITY": "-",
        ],
        defaultSettings: .recommended
    ),
    targets: [
        .target(
            name: "Rubick",
            destinations: [.mac],
            product: .app,
            bundleId: "com.fm.rubick",
            deploymentTargets: .macOS("13.0"),
            infoPlist: .extendingDefault(with: [
                "CFBundleDisplayName": "Rubick",
                "CFBundleShortVersionString": "0.1.0",
                "CFBundleVersion": "1",
                "LSMinimumSystemVersion": "13.0",
                "LSApplicationCategoryType": "public.app-category.productivity",
                "CFBundleIconFile": "AppIcon",
                // We don't ship a Storyboard or XIB — SwiftUI lifecycle.
                "NSMainStoryboardFile": "",
                "NSPrincipalClass": "NSApplication",
                "NSHumanReadableCopyright": "© BIGBALLON",
                // Subprocess management uses POSIX Process(); no
                // ScriptingBridge / AppleScript permission needed yet.
                // FSEvents-driven ingest is implemented in WatchService /
                // FSEventsMonitor (Settings → Watch mode).
                //
                // **Dev-mode** baked-in repo path: Tuist substitutes
                // ``$(SRCROOT)`` to the project root at compile time
                // so the .app can find ``backend/`` even when
                // launched via ``open -a`` (which sets cwd to ``/``).
                // Release builds will ship an embedded backend and
                // ignore this key.
                "RubickDevRepoRoot": "$(SRCROOT)",
            ]),
            sources: ["apps/Rubick/Sources/**"],
            resources: [
                .glob(
                    pattern: "apps/Rubick/Resources/**",
                    excluding: [
                        "apps/Rubick/Resources/nebula/**",
                        "apps/Rubick/Resources/.DS_Store",
                    ]
                ),
                // Nebula Three.js visualization (loaded by WKWebView)
                .folderReference(path: "apps/Rubick/Resources/nebula"),
            ],
            dependencies: []
        ),
    ]
)
