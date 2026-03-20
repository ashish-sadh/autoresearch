# Autoresearch — Live Training Log

This is a live log of an autonomous AI research experiment. A Claude agent runs in a loop, modifying a small language model's architecture and training setup in 5-minute experiments on a Mac (Apple M5 Max, 64GB unified memory). Changes that lower validation bits-per-byte (val_bpb — lower is better) are kept; others are discarded.

Every 5 improvements, the best architecture found so far is scaled up to depth=24 (~400M parameters) and trained for 1 hour on web text (climbmix-400b, a 400B token dataset), then fine-tuned for chat using SmolTalk instruction data. Each entry below marks one of those deep-train milestones.

The explore loop runs at depth=4 (small and fast — ~5M params, ~500 experiments/day). The deep-train runs at depth=24 (large and slow — ~400M params, 1 hour each). Architectural improvements found at small scale transfer to the large model.

**Dataset**: climbmix-400b web text (300 shards, ~18B tokens downloaded)
**Hardware**: Apple M5 Max, 64GB unified memory, macOS MPS
**Optimizer**: Muon (matrix params) + AdamW (embeddings/scalars)

---

*Entries below are added automatically after each deep-train. Newest at the bottom.*

---

## #1 · 2026-03-20 · 1.0h accumulated pretraining

**val_bpb**: 1.138994 · **model**: depth=24 · ~52M params

**Last 5 improvements**: MATRIX_LR 0.060→0.065 gave marginal gain at d4 256-dim; removing weight decay (WD 0.1→0.0) was a surprisingly large win — small models with short training horizons don't benefit from regularization; EMBEDDING_LR 0.6→0.4 improved further in the no-WD regime; MATRIX_LR 0.065→0.070 confirmed the optimum shifts higher without weight decay; SCALAR_LR 1.0→2.0 squeezed out one more improvement, showing per-layer scalars adapt faster in the unregularized regime.

**Q: Why is the sky blue?**
> The 350+ Creation of a Quartz Series, also known as AVC, is an innovative and exciting product that combines stunning scenes with vibrant colors. Its colorful leaves captivate the eyes of its readers, adding depth to your visual interest. When viewed from above, it's time to explore various vivid and dramatic shadows. The sky blue sets the tone for a vibrant and stunning effect on the scene, making it a must-have in any professional setting.

---
