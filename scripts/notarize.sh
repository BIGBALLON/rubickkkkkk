#!/usr/bin/env bash
# scripts/notarize.sh — Developer-ID sign + notarize + staple a Rubick DMG.
#
# Apple's Gatekeeper trusts a downloaded DMG only when:
#
#   1. Every Mach-O inside the .app is signed with a valid
#      Developer ID Application identity, with the hardened runtime
#      enabled and the right entitlements.
#   2. The signed bundle is submitted to Apple via ``notarytool``,
#      which scans for malware + verifies the signature, and Apple
#      issues a "ticket" for it.
#   3. The ticket is stapled into the DMG so Gatekeeper can verify
#      offline (without contacting Apple at first-open).
#
# This script handles all three. It expects three environment
# variables — none are committed to the repo or read from any file
# by accident:
#
#   APPLE_ID                       Apple ID email (e.g. you@apple.com)
#   APPLE_TEAM_ID                  10-character Team ID from
#                                  https://developer.apple.com/account
#                                  (Membership → Team ID)
#   APPLE_APP_SPECIFIC_PASSWORD    App-specific password generated at
#                                  https://appleid.apple.com (Sign-In
#                                  & Security → App-Specific Passwords)
#
# Plus the signing identity name (what ``security find-identity -p
# codesigning`` calls it):
#
#   APPLE_SIGNING_IDENTITY        e.g. "Developer ID Application:
#                                  Your Name (TEAMID12345)"
#                                  defaults to: the first
#                                  ``Developer ID Application`` row
#                                  in your keychain.
#
# Usage::
#
#     ./scripts/notarize.sh build/Rubick-0.1.0.dmg
#
# Pipeline (six steps), each a separate ``echo``+command pair so a
# failure tells you exactly which step blew up:
#
#   1. Resolve & sanity-check the input DMG.
#   2. Mount the DMG and copy ``Rubick.app`` to a temp work dir.
#   3. Re-sign the .app with the Developer ID identity, hardened
#      runtime on, with the entitlements needed for our subprocess
#      + Carbon hotkey + filesystem access.
#   4. Re-pack a fresh DMG around the signed .app (so the signature
#      is what users actually receive).
#   5. ``notarytool submit --wait`` → upload + wait for Apple's
#      verdict (typically 1-5 minutes).
#   6. ``stapler staple`` the verdict into the DMG so Gatekeeper
#      can verify offline.

set -euo pipefail

if (( $# < 1 )); then
    echo "usage: $0 path/to/Rubick.dmg" >&2
    exit 2
fi
DMG_IN="$1"
[[ -f "$DMG_IN" ]] || { echo "not a file: $DMG_IN" >&2; exit 1; }

: "${APPLE_ID:?APPLE_ID env var must be set}"
: "${APPLE_TEAM_ID:?APPLE_TEAM_ID env var must be set}"
: "${APPLE_APP_SPECIFIC_PASSWORD:?APPLE_APP_SPECIFIC_PASSWORD env var must be set}"

# Pick the signing identity. Allow override via env; otherwise grab
# the first Developer ID Application row from the keychain.
if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    APPLE_SIGNING_IDENTITY="$(
        security find-identity -v -p codesigning \
        | awk -F'"' '/Developer ID Application/{print $2; exit}'
    )"
fi
[[ -n "$APPLE_SIGNING_IDENTITY" ]] || {
    echo "no Developer ID Application identity in your keychain" >&2
    echo "see: https://developer.apple.com/help/account/create-certificates/create-a-certificate-signing-request/" >&2
    exit 1
}

WORK_DIR="$(mktemp -d /tmp/rubick-notarize.XXXXXX)"
trap "rm -rf '$WORK_DIR'" EXIT

# 1. preflight already done above.
echo "=== input DMG: $DMG_IN ==="
echo "    signing identity: $APPLE_SIGNING_IDENTITY"

# 2. mount + extract.
echo "=== mount + extract Rubick.app ==="
MOUNT_POINT="$WORK_DIR/mount"
hdiutil attach -nobrowse -mountpoint "$MOUNT_POINT" "$DMG_IN" >/dev/null
APP_SRC="$MOUNT_POINT/Rubick.app"
[[ -d "$APP_SRC" ]] || { hdiutil detach -quiet "$MOUNT_POINT"; echo "Rubick.app not found inside DMG" >&2; exit 1; }
cp -R "$APP_SRC" "$WORK_DIR/Rubick.app"
hdiutil detach -quiet "$MOUNT_POINT"

# 3. real Developer ID sign with hardened runtime.
ENTITLEMENTS="$WORK_DIR/entitlements.plist"
cat >"$ENTITLEMENTS" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- We spawn a Python subprocess for the backend. Without these,
         the hardened runtime kills the Process() launch with EPERM. -->
    <key>com.apple.security.cs.allow-jit</key><true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
    <key>com.apple.security.cs.disable-library-validation</key><true/>
    <!-- File access — we walk user-chosen folders. Notarization is
         compatible with non-sandboxed apps that declare these. -->
    <key>com.apple.security.files.user-selected.read-write</key><true/>
</dict>
</plist>
PLIST

echo "=== codesign --deep --options runtime ==="
codesign --force --deep --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" \
    --sign "$APPLE_SIGNING_IDENTITY" \
    "$WORK_DIR/Rubick.app"
codesign --verify --strict --verbose=2 "$WORK_DIR/Rubick.app"

# 4. re-pack DMG around the signed app.
DMG_OUT="$(dirname "$DMG_IN")/$(basename "$DMG_IN" .dmg)-signed.dmg"
rm -f "$DMG_OUT"
echo "=== re-pack DMG: $DMG_OUT ==="
create-dmg \
    --volname "Rubick" \
    --window-pos 200 120 \
    --window-size 540 360 \
    --icon-size 100 \
    --icon "Rubick.app" 140 180 \
    --hide-extension "Rubick.app" \
    --app-drop-link 400 180 \
    --no-internet-enable \
    "$DMG_OUT" \
    "$WORK_DIR/Rubick.app" \
    | tail -3

# 5. submit + wait.
echo "=== notarytool submit --wait ==="
xcrun notarytool submit "$DMG_OUT" \
    --apple-id "$APPLE_ID" \
    --team-id "$APPLE_TEAM_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --wait

# 6. staple the ticket so first-open works without the Apple round-trip.
echo "=== stapler staple ==="
xcrun stapler staple "$DMG_OUT"
xcrun stapler validate "$DMG_OUT"

DMG_SIZE="$(du -h "$DMG_OUT" | awk '{print $1}')"
DMG_SHA="$(shasum -a 256 "$DMG_OUT" | awk '{print $1}')"

echo
echo "✔ Notarized & stapled: $DMG_OUT ($DMG_SIZE)"
echo "  sha256: $DMG_SHA"
echo
echo "Ready for distribution. Gatekeeper accepts this on first open"
echo "without a network round-trip thanks to the stapled ticket."
