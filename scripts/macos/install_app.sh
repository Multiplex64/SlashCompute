#!/bin/bash
# Write ~/Applications/compute.app (or --dest) that launches this checkout's venv.
set -euo pipefail

REPO=""
DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    *) echo "usage: $0 [--repo PATH] [--dest PATH]" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "$REPO" ]]; then
  REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
REPO="$(cd "$REPO" && pwd)"

if [[ -z "$DEST" ]]; then
  mkdir -p "$HOME/Applications"
  DEST="$HOME/Applications/compute.app"
fi

PY="$REPO/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No venv at $PY — run: cd \"$REPO\" && uv sync --group dev" >&2
  exit 1
fi

CONTENTS="$DEST/Contents"
MACOS="$CONTENTS/MacOS"
RES="$CONTENTS/Resources"
rm -rf "$DEST"
mkdir -p "$MACOS" "$RES"

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>/compute</string>
  <key>CFBundleExecutable</key><string>compute</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>com.slashcompute.app</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>compute</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

cat > "$MACOS/compute" <<EOF
#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
cd "$REPO"
exec "$PY" -m slashcompute.launcher.main
EOF
chmod +x "$MACOS/compute"

python3 - "$RES" <<'PY'
import struct, sys, zlib
from pathlib import Path

out = Path(sys.argv[1])
size = 1024
# carbon field, paper slash
px = bytearray()
for y in range(size):
    px.append(0)
    for x in range(size):
        # two stacked bars like / 
        t = (x + (size - 1 - y)) / (size * 2)
        band = abs((x - y) - size * 0.18) < size * 0.07
        if band:
            px.extend((0xF2, 0xF2, 0xF0))
        else:
            px.extend((0x14, 0x14, 0x14))

def chunk(tag, data):
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

raw = bytes(px)
ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
(out / "icon.png").write_bytes(png)
PY

if command -v sips >/dev/null && command -v iconutil >/dev/null; then
  ICONSET="$RES/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for pair in 16:16 32:16 32:32 64:32 128:128 256:128 256:256 512:256 512:512 1024:512; do
    px="${pair%%:*}"
    base="${pair##*:}"
    if [[ "$px" == "$base" ]]; then
      name="icon_${base}x${base}.png"
    else
      name="icon_${base}x${base}@2x.png"
    fi
    sips -z "$px" "$px" "$RES/icon.png" --out "$ICONSET/$name" >/dev/null
  done
  iconutil -c icns -o "$RES/AppIcon.icns" "$ICONSET"
  rm -rf "$ICONSET"
fi

echo "Installed $DEST"
echo "Double-click /compute, or open -a $DEST"
