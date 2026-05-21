# Contributing

Open an issue before large PRs. **English only** for UI strings, logs, and code comments.

## Setup

```bash
cd backend && uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

brew install --cask tuist
tuist generate
```

Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md).

## Run locally

```bash
cd backend && pytest && ruff check src tests

xcodebuild -workspace Rubick.xcworkspace -scheme Rubick \
  -configuration Debug -destination 'platform=macOS,arch=arm64' build CODE_SIGNING_ALLOWED=NO

APP=$(find ~/Library/Developer/Xcode/DerivedData/Rubick-*/Build/Products/Debug \
        -maxdepth 2 -name 'Rubick.app' | head -1)
RUBICK_DATA_DIR="$HOME/Library/Application Support/RubickDev" open -a "$APP"
```

**Slow tests** (loads ~1.8 GB MLX model): `RUBICK_RUN_SLOW=1 pytest`  
**Fused e2e** needs a local fixture tree — see [backend/tests/fixtures/fused_e2e/README.md](backend/tests/fixtures/fused_e2e/README.md).

Tear down scratch data: `rm -rf ~/Library/Application\ Support/RubickDev`

## Code & comments

- Describe **what the code does now**, not obsolete roadmap stages, external spec section numbers, or pointers to planning docs (`PICKUP.md`, `idea/specs/`, etc.).
- Prefer linking to [ARCHITECTURE.md](ARCHITECTURE.md) for cross-cutting behaviour (retrieval, process model, data layout).
- Keep comments short; delete stale notes instead of annotating history inline (git history exists).
- Python: `ruff check` must pass. Match surrounding module style.
- Swift: follow existing patterns in `apps/Rubick/Sources/`.

## Commits

Imperative mood, area prefix: `feat(retrieve): …`, `fix(app): …`

## Release build (optional)

```bash
./scripts/build_dmg.sh              # full Release + backend bundle + DMG
./scripts/build_dmg.sh --skip-build --skip-bundle   # reuse existing artifacts
```

Version comes from `Project.swift` → `CFBundleShortVersionString`.

## License

Copyright © BIGBALLON. MIT licensed — see [LICENSE](LICENSE). Do not commit model weights. The embedding model is CC BY-NC 4.0 — see [README.md](README.md).

Be kind.
