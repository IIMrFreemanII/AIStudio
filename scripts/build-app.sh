#!/bin/zsh
# Builds AIStudio.app into ./build. Usage: scripts/build-app.sh [Debug|Release]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-Release}
DERIVED=.build/xcode
xcodebuild build -scheme AIStudioApp -destination 'platform=macOS' -derivedDataPath "$DERIVED" \
  -configuration "$CONFIG" -skipPackagePluginValidation -skipMacroValidation -quiet
PRODUCTS="$DERIVED/Build/Products/$CONFIG"
APP=build/AIStudio.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PRODUCTS/AIStudioApp" "$APP/Contents/MacOS/AIStudio"
# SwiftPM resource bundles (MLX's Metal library, tokenizer data…) are found via the main bundle's Resources.
for b in "$PRODUCTS"/*.bundle; do cp -R "$b" "$APP/Contents/Resources/"; done
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>AIStudio</string>
  <key>CFBundleDisplayName</key><string>AIStudio</string>
  <key>CFBundleIdentifier</key><string>local.aistudio</string>
  <key>CFBundleExecutable</key><string>AIStudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.3</string>
  <key>CFBundleVersion</key><string>3</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
# Signing: ad hoc by default, which gives every build a new identity, so macOS asks again for Keychain access (the
# Hugging Face token) after each rebuild. With a stable identity, "Always Allow" sticks: create a self-signed
# code-signing certificate named "AIStudio Local Signing" in Keychain Access (Certificate Assistant › Create a
# Certificate…, type Code Signing), or set AISTUDIO_SIGN_IDENTITY to another identity's name. A certificate made
# before the rename ("MLXComfyUI Local Signing") is used too.
IDENTITY=${AISTUDIO_SIGN_IDENTITY:-${MLXCOMFY_SIGN_IDENTITY:-}}
if [[ -z "$IDENTITY" ]]; then
  IDENTITIES="$(security find-identity -p codesigning 2>/dev/null || true)"
  for name in "AIStudio Local Signing" "MLXComfyUI Local Signing"; do
    if grep -q "\"$name\"" <<<"$IDENTITIES"; then IDENTITY="$name"; break; fi
  done
fi
if [[ -n "$IDENTITY" ]]; then
  codesign --force --deep --sign "$IDENTITY" "$APP" >/dev/null 2>&1 || echo "warning: signing with “$IDENTITY” failed"
else
  codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "warning: ad-hoc signing failed"
fi
echo "built $APP"
