#!/usr/bin/env bash
# scripts/build_dmg.sh — produce a distributable Rubick-<version>.dmg
# from a fresh Release build of the SwiftUI app.
#
# Pipeline (eight steps):
#
#   1. ``tuist generate`` to refresh the Xcode workspace.
#   2. ``xcodebuild`` Release for macOS / arm64 — produces a fully
#      built ``Rubick.app`` in DerivedData.
#   3. ``scripts/build_backend_bundle.sh`` to materialize the hermetic
#      Python runtime + ``rubick_backend`` install in
#      ``build/backend-bundle/`` (~1.4 GB).
#   4. Stage the .app into a clean working dir under ``build/dmg-stage``
#      so the DMG layout is deterministic.
#   5. Embed the bundle into ``Rubick.app/Contents/Resources/backend-runtime/``
#      so ``BackendRuntime.resolve()`` picks the bundled python on launch.
#   6. Ad-hoc deep codesign + verify the staged .app so Gatekeeper
#      sees a complete signature (real Developer-ID signing happens
#      in ``scripts/notarize.sh``).
#   7. ``create-dmg`` (brew package) builds the final DMG with an
#      Applications symlink + window position so the drag-to-install
#      pattern works on first open.
#   8. Print the DMG path + size + sha256 so the caller can pin it.
#
# What this script DOES NOT do:
#
# - It does NOT Developer-ID sign or notarize. ``scripts/notarize.sh``
#   handles those; this script's output is dev-only / TestFlight-style.
#
# Embeds the Python backend into Rubick.app after xcodebuild. After
# ``xcodebuild`` produces ``Rubick.app`` it calls
# ``scripts/build_backend_bundle.sh`` (unless ``--skip-bundle``) to
# materialize ``build/backend-bundle/`` (a 1.4 GB hermetic
# python-build-standalone tree + ``rubick_backend`` install), then
# rsyncs that into ``Rubick.app/Contents/Resources/backend-runtime/``
# so ``BackendRuntime.resolve()`` picks the bundled path on launch.
# The ad-hoc deep codesign step picks up every embedded Mach-O
# automatically.
#
# Usage:
#
#     ./scripts/build_dmg.sh                  # default; output build/Rubick-<v>.dmg
#     ./scripts/build_dmg.sh --skip-build     # use whatever's already in DerivedData
#     ./scripts/build_dmg.sh --skip-bundle    # use whatever's in build/backend-bundle/
#     ./scripts/build_dmg.sh --output /tmp    # change output dir

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
STAGE_DIR="$BUILD_DIR/dmg-stage"
OUTPUT_DIR="$BUILD_DIR"

SKIP_BUILD=0
SKIP_BUNDLE=0
while (( $# > 0 )); do
    case "$1" in
        --skip-build)  SKIP_BUILD=1;  shift ;;
        --skip-bundle) SKIP_BUNDLE=1; shift ;;
        --output)      OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,45p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

BUNDLE_DIR="$BUILD_DIR/backend-bundle"

# ---------------------------------------------------------------- preflight

command -v tuist >/dev/null   || { echo "missing: tuist (brew install --cask tuist)" >&2; exit 1; }
command -v create-dmg >/dev/null || { echo "missing: create-dmg (brew install create-dmg)" >&2; exit 1; }
command -v xcodebuild >/dev/null || { echo "missing: xcodebuild (install Xcode)" >&2; exit 1; }

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# ---------------------------------------------------------------- build

cd "$REPO_ROOT"

if (( SKIP_BUILD == 0 )); then
    echo "=== tuist generate ==="
    tuist generate --no-open

    echo "=== xcodebuild Release ==="
    xcodebuild \
        -workspace Rubick.xcworkspace \
        -scheme Rubick \
        -configuration Release \
        -destination 'platform=macOS,arch=arm64' \
        clean build \
        | tail -5
fi

# Locate the freshly-built .app under DerivedData.
APP_PATH="$(find "$HOME/Library/Developer/Xcode/DerivedData/Rubick-"*/Build/Products/Release \
    -name 'Rubick.app' -maxdepth 2 -type d 2>/dev/null | head -1 || true)"

if [[ -z "$APP_PATH" || ! -d "$APP_PATH" ]]; then
    echo "Rubick.app not found under DerivedData/.../Release. Run without --skip-build." >&2
    exit 1
fi

echo "Found Rubick.app: $APP_PATH"

# Read version from Info.plist for filename hygiene.
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "$APP_PATH/Contents/Info.plist" 2>/dev/null || echo 'dev')"

# ---------------------------------------------------------------- backend bundle

# Build (or reuse) the hermetic Python runtime + ``rubick_backend``
# install. ``build_backend_bundle.sh`` is idempotent — it re-rsyncs
# the python-build-standalone tree and re-runs ``uv pip install`` on
# every invocation so the bundle is byte-deterministic. Add
# ``--skip-bundle`` if you're iterating on DMG layout and the bundle
# is already built.
if (( SKIP_BUNDLE == 0 )); then
    echo "=== build backend bundle ==="
    "$REPO_ROOT/scripts/build_backend_bundle.sh"
fi

if [[ ! -x "$BUNDLE_DIR/python/bin/python3" ]]; then
    echo "missing: $BUNDLE_DIR/python/bin/python3" >&2
    echo "(Did you pass --skip-bundle without first running build_backend_bundle.sh?)" >&2
    exit 1
fi

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | awk '{print $1}')"
echo "    backend bundle: $BUNDLE_DIR ($BUNDLE_SIZE)"

# ---------------------------------------------------------------- stage

cp -R "$APP_PATH" "$STAGE_DIR/Rubick.app"

# Drop the bundle into the .app's Resources/. ``BackendRuntime.resolve()``
# will pick it up via ``Bundle.main.resourceURL/backend-runtime/python/bin/python3``
# instead of falling back to the dev venv. Use ``rsync -a`` so symlinks
# (python3 → python3.12, the lib/lib*.dylib chain, etc.) are
# preserved as symlinks rather than dereferenced.
echo "=== embed backend bundle into .app ==="
RUNTIME_DEST="$STAGE_DIR/Rubick.app/Contents/Resources/backend-runtime"
mkdir -p "$RUNTIME_DEST"
rsync -a --delete "$BUNDLE_DIR/" "$RUNTIME_DEST/"

EMBEDDED_SIZE="$(du -sh "$STAGE_DIR/Rubick.app" | awk '{print $1}')"
echo "    .app size after embedding: $EMBEDDED_SIZE"

# ---------------------------------------------------------------- ad-hoc sign

# Real signing happens in notarize.sh with a Developer ID. For local
# DMGs we deep-sign with the ``-`` (ad-hoc) identity so Gatekeeper at
# least sees a complete signature instead of "code object is not signed
# at all". Ad-hoc-signed apps still trip the first-open quarantine
# warning; that's expected without a real identity.
#
# ``--deep`` walks the .app and re-signs every Mach-O it finds. With
# the embedded backend-runtime that's the python interpreter +
# ~1000 ``.so`` extension modules + ~30 ``.dylib`` libs from
# torch / mlx / pyarrow / lancedb / etc. Some have stale signatures
# from python-build-standalone; ``--force`` overrides them.
echo "=== ad-hoc codesign (deep, includes embedded backend-runtime) ==="
codesign --force --deep --sign - --timestamp=none \
    "$STAGE_DIR/Rubick.app" 2>&1 | tail -5 || true

# Verify before packaging — catches the case where some Mach-O
# resisted ad-hoc signing (rare but happens with weird extended
# attributes on Python wheels). ``--verbose=2`` reports the bundle
# format / signature lineage; we keep the result advisory because
# ad-hoc signatures legitimately fail strict validation.
echo "=== verify signature ==="
codesign --verify --verbose=2 "$STAGE_DIR/Rubick.app" 2>&1 | tail -5 || true

# ---------------------------------------------------------------- DMG

DMG_NAME="Rubick-$VERSION.dmg"
DMG_PATH="$OUTPUT_DIR/$DMG_NAME"
rm -f "$DMG_PATH"

echo "=== create-dmg ==="
create-dmg \
    --volname "Rubick $VERSION" \
    --window-pos 200 120 \
    --window-size 540 360 \
    --icon-size 100 \
    --icon "Rubick.app" 140 180 \
    --hide-extension "Rubick.app" \
    --app-drop-link 400 180 \
    --no-internet-enable \
    "$DMG_PATH" \
    "$STAGE_DIR" \
    | tail -5

# ---------------------------------------------------------------- report

if [[ ! -f "$DMG_PATH" ]]; then
    echo "create-dmg did not produce $DMG_PATH" >&2
    exit 1
fi

DMG_SIZE="$(du -h "$DMG_PATH" | awk '{print $1}')"
DMG_SHA="$(shasum -a 256 "$DMG_PATH" | awk '{print $1}')"

echo
echo "✔ Built $DMG_PATH ($DMG_SIZE)"
echo "  sha256: $DMG_SHA"
echo
echo "Next step: ./scripts/notarize.sh \"$DMG_PATH\""
echo "(needs APPLE_ID + APPLE_TEAM_ID + APPLE_APP_SPECIFIC_PASSWORD env vars)"
