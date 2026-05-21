#!/usr/bin/env bash
# scripts/build_backend_bundle.sh — build a self-contained Python
# runtime + ``rubick_backend`` install for embedding in ``Rubick.app``.
#
# Pipeline (five steps):
#
#   1. ``rsync`` ``python-build-standalone`` (cpython-3.12 macOS arm64)
#      out of the local uv cache into ``build/backend-bundle/python/``.
#      That distribution is engineered for relocation: the interpreter
#      only links ``/System`` + ``/usr/lib``, no Homebrew, no rpath
#      surgery needed.
#   2. ``uv pip install`` ``backend/`` against the bundled python so all
#      prod deps (mlx / lancedb / fastapi / pillow / …) AND
#      ``rubick_backend`` itself land in the bundled
#      ``site-packages``. No build isolation, no caching — we want a
#      deterministic image of the deps tree. (As of the v1.x torch
#      removal, ``torch`` / ``torchvision`` / ``transformers`` are
#      dev-only and not installed into the bundle; image / video
#      preprocessing is hand-rolled in PIL + numpy.)
#   3. Prune ``__pycache__`` / ``*.pyc`` / ``*.pyo`` (regenerated on
#      first import; bloats the bundle). Stay conservative — do NOT
#      strip stdlib ``test`` (already absent) or third-party ``tests/``
#      packages (some libs import them at runtime).
#   4. Smoke test: have the bundled python ``import`` the heavy modules
#      (``rubick_backend.embed.loader`` triggers MLX, ``store.schema``
#      pulls LanceDB). If this fails, the bundle is unusable.
#   5. Write ``VERSION`` + ``SUMMARY.md`` so reviewers / future-you can
#      see exactly what python-build-standalone version + which
#      ``rubick_backend`` source produced the artifact, and which
#      packages dominate the size.
#
# What this script DOES NOT do:
#
# - It does NOT Developer-ID sign or notarize (see ``scripts/notarize.sh``).
# - It does NOT ship a fresh ``python-build-standalone`` download by
#   default — it copies from the local uv cache. Use ``--source-python``
#   to point at a different tree.
#
# The bundle is embedded into ``Rubick.app`` by ``scripts/build_dmg.sh``
# and picked up at launch by ``PythonProcess`` / ``BackendRuntime``.
#
# Usage:
#
#     ./scripts/build_backend_bundle.sh
#         # default; output build/backend-bundle/
#     ./scripts/build_backend_bundle.sh --output /tmp/bundle
#         # change output dir
#     ./scripts/build_backend_bundle.sh --source-python /path/to/cpy
#         # use a different python-build-standalone tree
#     ./scripts/build_backend_bundle.sh --skip-deps
#         # skip uv pip install (smoke + report on existing bundle)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/build/backend-bundle"
SOURCE_PYTHON="${RUBICK_BUNDLE_SOURCE_PYTHON:-$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none}"
SKIP_DEPS=0

while (( $# > 0 )); do
    case "$1" in
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --source-python) SOURCE_PYTHON="$2"; shift 2 ;;
        --skip-deps)     SKIP_DEPS=1; shift ;;
        -h|--help)       sed -n '2,55p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- preflight

if [[ ! -d "$SOURCE_PYTHON" ]]; then
    cat >&2 <<EOF
build_backend_bundle.sh: source python not found at:
  $SOURCE_PYTHON

Either:
  * Set RUBICK_BUNDLE_SOURCE_PYTHON to a python-build-standalone install
    (the dev ``backend/.venv`` already uses one — see
    ``readlink backend/.venv/bin/python`` to find its prefix), or
  * Pass --source-python /path/to/cpython-3.12-macos-aarch64-none
EOF
    exit 1
fi

if [[ ! -x "$SOURCE_PYTHON/bin/python3" ]]; then
    echo "no executable python3 under $SOURCE_PYTHON/bin/" >&2
    exit 1
fi

command -v rsync >/dev/null || { echo "missing: rsync" >&2; exit 1; }
command -v uv >/dev/null    || { echo "missing: uv (brew install uv)" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT_DIR")"

# ---------------------------------------------------------------- 1. python copy

if (( SKIP_DEPS == 0 )); then
    echo "=== rsync python-build-standalone ==="
    echo "    src: $SOURCE_PYTHON"
    echo "    dst: $OUTPUT_DIR/python/"

    rm -rf "$OUTPUT_DIR/python"
    mkdir -p "$OUTPUT_DIR/python"

    # ``--exclude=BUILD`` drops the python-build-standalone build
    # metadata file (~4 KB) — useful for reproducibility tracking but
    # not needed at runtime. Everything else stays so relocation is a
    # straight copy.
    rsync -a \
        --exclude='BUILD' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        "$SOURCE_PYTHON/" "$OUTPUT_DIR/python/"

    # python-build-standalone ships a PEP 668 ``EXTERNALLY-MANAGED``
    # marker so that ``uv``/``pip`` users can't accidentally mutate
    # the python copy that ``uv`` keeps in its global cache. For our
    # bundle that's exactly what we *want* — this is a private copy,
    # not the cache, and the next step is literally ``uv pip install``
    # into it. Drop the marker (only on our copy; the cache is
    # untouched).
    rm -f "$OUTPUT_DIR/python/lib/python3.12/EXTERNALLY-MANAGED"
fi

BUNDLE_PYTHON="$OUTPUT_DIR/python/bin/python3"
if [[ ! -x "$BUNDLE_PYTHON" ]]; then
    echo "bundled python missing after rsync: $BUNDLE_PYTHON" >&2
    exit 1
fi

PY_VER="$("$BUNDLE_PYTHON" -c 'import sys; print(sys.version.split()[0])')"
echo "    bundled python: $PY_VER"

# ---------------------------------------------------------------- 2. install deps

if (( SKIP_DEPS == 0 )); then
    echo
    echo "=== uv pip install backend/ (prod deps + rubick_backend) ==="

    # ``--no-cache`` is intentional: the bundle should be byte-identical
    # whether or not the developer has a uv cache primed. This costs
    # ~30s extra per build but removes a whole class of "works on my
    # machine" bugs. ``--reinstall`` is similarly defensive.
    uv pip install \
        --python "$BUNDLE_PYTHON" \
        --no-cache \
        --reinstall \
        "$REPO_ROOT/backend"
fi

# ---------------------------------------------------------------- 3. prune

echo
echo "=== prune ==="

SP="$OUTPUT_DIR/python/lib/python3.12/site-packages"
PRUNED_BYTES_BEFORE=$(du -sk "$SP" 2>/dev/null | awk '{print $1}')

# (a) Bytecode caches. uv installs without compiling .pyc, so there
# usually aren't any after the install step, but a smoke run earlier
# in this script may have produced some. Python regenerates these on
# first import.
find "$SP" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$SP" \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true

# (b) C/C++ headers. mlx / pyarrow / numpy etc. all ship an
# ``include/`` subtree so downstream packages can build C extensions
# against them at install time. Once the bundle is sealed, no new
# extensions are ever built — these are pure dead weight.
for hdr in \
    "$SP/mlx/include" \
    "$SP/pyarrow/include" \
    "$SP/numpy/_core/include" \
    "$SP/numpy/core/include"
do
    [[ -d "$hdr" ]] && rm -rf "$hdr"
done

# (c) ``*.dist-info/RECORD`` files. Regenerated by pip on demand;
# absent for several MB across hundreds of packages.
find "$SP" -path '*.dist-info/RECORD' -delete 2>/dev/null || true

# What we *don't* prune (because the smoke test caught these as
# unsafe earlier):
#
# - ``pyarrow/libarrow_{flight,substrait,acero,dataset}.*.dylib``.
#   pyarrow's ``lib.cpython-312-darwin.so`` links every sibling at
#   the dyld level — removing any one of them surfaces as a
#   ``Library not loaded: @rpath/libarrow_substrait...`` ImportError
#   inside ``rubick_backend.store.schema``. ~25 MB stays.

PRUNED_BYTES_AFTER=$(du -sk "$SP" 2>/dev/null | awk '{print $1}')
PRUNED_DELTA_MB=$(( (PRUNED_BYTES_BEFORE - PRUNED_BYTES_AFTER) / 1024 ))
echo "    site-packages: $((PRUNED_BYTES_BEFORE / 1024)) MB → $((PRUNED_BYTES_AFTER / 1024)) MB (-${PRUNED_DELTA_MB} MB)"

# ---------------------------------------------------------------- 4. smoke

echo
echo "=== smoke test ==="

# Touch every heavyweight import path the backend exercises at startup.
# If any of these fail, the bundle is broken — usually because uv
# resolved a wheel for the host python instead of the bundled one.
"$BUNDLE_PYTHON" - <<'PY'
import sys, importlib

print(f"  python:  {sys.version.split()[0]}  ({sys.executable})")
for mod in [
    "rubick_backend",
    "rubick_backend.main",
    "rubick_backend.embed.loader",       # imports mlx (~189 MB native)
    "rubick_backend.embed.preprocessing",  # PIL + numpy (no torch)
    "rubick_backend.store.schema",       # imports lancedb + pyarrow
    "rubick_backend.ingest.image",       # imports PIL + pillow_heif
    "rubick_backend.ingest.video",       # imports av (PyAV)
    # Direct touches for the prune step's blast radius.
    "pyarrow",                           # ``flight/substrait/acero`` kept
    "lancedb",                           # depends on pyarrow.compute
]:
    importlib.import_module(mod)
    print(f"  ok:      {mod}")
print("  all heavy imports resolved against the bundled python")
PY

# ---------------------------------------------------------------- 5. report

echo
echo "=== bundle report ==="

TOTAL_BYTES=$(du -sk "$OUTPUT_DIR" | awk '{print $1*1024}')
TOTAL_HUMAN=$(du -sh "$OUTPUT_DIR" | awk '{print $1}')
PY_HUMAN=$(du -sh "$OUTPUT_DIR/python" | awk '{print $1}')
SP_HUMAN=$(du -sh "$OUTPUT_DIR/python/lib/python3.12/site-packages" | awk '{print $1}')

echo "    output:        $OUTPUT_DIR"
echo "    total:         $TOTAL_HUMAN ($TOTAL_BYTES bytes)"
echo "    python tree:   $PY_HUMAN"
echo "    site-packages: $SP_HUMAN"
echo
echo "    top 15 packages by size:"
du -sh "$OUTPUT_DIR/python/lib/python3.12/site-packages"/* 2>/dev/null \
    | sort -h | tail -15 | awk '{printf "      %-8s  %s\n", $1, $2}'

# ---------------------------------------------------------------- 6. metadata

cat > "$OUTPUT_DIR/VERSION" <<EOF
python-build-standalone-source: $SOURCE_PYTHON
python-version:                 $PY_VER
host:                           $(uname -srm)
built:                          $(date -u +%Y-%m-%dT%H:%M:%SZ)
rubick_backend-source:          $REPO_ROOT/backend
EOF

# Top-15 list snapshot for the markdown summary (regenerate so the
# numbers in SUMMARY.md match what we just printed above).
TOP15=$(du -sh "$OUTPUT_DIR/python/lib/python3.12/site-packages"/* 2>/dev/null \
    | sort -h | tail -15 | awk '{printf "%-8s  %s\n", $1, $2}')

cat > "$OUTPUT_DIR/SUMMARY.md" <<EOF
# Rubick backend bundle

Generated by \`scripts/build_backend_bundle.sh\` on $(date -u +%Y-%m-%d).

| Metric | Value |
|---|---|
| Total | $TOTAL_HUMAN |
| Python tree | $PY_HUMAN |
| site-packages | $SP_HUMAN |
| python-build-standalone | \`$(basename "$SOURCE_PYTHON")\` |
| python | $PY_VER |

## Top 15 packages by size

\`\`\`
$TOP15
\`\`\`

## Next step

Copy into a Release ``Rubick.app`` via ``./scripts/build_dmg.sh`` (or
reuse an existing bundle with ``--skip-bundle``).
EOF

echo
echo "    metadata: $OUTPUT_DIR/VERSION + SUMMARY.md"
echo
echo "Done."
