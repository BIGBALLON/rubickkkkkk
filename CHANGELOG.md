# Changelog

## [0.1.0] — 2026-05-21

### Added

- Local multimodal search (text, image, short video ≤2 min) on Apple Silicon via MLX
- Pulsar quick panel (`⌥+Space`) and Chronosphere 3D map (`⌥+Space×2`)
- Hybrid retrieval: vector ANN + BM25 → RRF → doc fold; fused text+image queries
- Watched-folder indexing with scheduled / live / manual rescan modes
- Hermetic Python backend bundle + optional DMG build scripts
- Pure MLX inference — no PyTorch / transformers at runtime (hand-rolled preprocess)
