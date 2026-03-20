# Autoresearch — Live Training Log

This is a live log of an autonomous AI research experiment. A Claude agent runs in a loop, modifying a small language model's architecture and training setup in 5-minute experiments on a Mac (Apple M5 Max, 64GB unified memory). Changes that lower validation bits-per-byte (val_bpb — lower is better) are kept; others are discarded.

Every 5 improvements, a deep-train is triggered at depth=24 (~860M parameters, 1536-dim) and trained for 1 hour on web text (climbmix-400b, a 400B token dataset), then fine-tuned for chat using SmolTalk instruction data. Each entry below marks one of those deep-train milestones.

The explore loop runs at a small, fast depth (typically ~5M params, ~500 experiments/day). The deep-train runs at fixed depth=24 (~860M params, 1536-dim, 1 hour each). The large model trains from scratch or resumes its own accumulated checkpoint — it doesn't transfer weights from the small model, but benefits from hyperparameter insights found in the explore loop.

**Dataset**: climbmix-400b web text (300 shards, ~18B tokens downloaded)
**Hardware**: Apple M5 Max, 64GB unified memory, macOS MPS
**Optimizer**: Muon (matrix params) + AdamW (embeddings/scalars)

---

*Entries below are added automatically after each deep-train. Newest at the bottom.*

---

## #1 · 2026-03-20 · 1.0h accumulated pretraining

**val_bpb**: 1.138994 · **model**: depth=24, 256-dim · ~52M params *(note: early run used small architecture; future runs scale to 1536-dim ~860M params)*

**Last 5 improvements**: MATRIX_LR 0.060→0.065 gave marginal gain at d4 256-dim; removing weight decay (WD 0.1→0.0) was a surprisingly large win — small models with short training horizons don't benefit from regularization; EMBEDDING_LR 0.6→0.4 improved further in the no-WD regime; MATRIX_LR 0.065→0.070 confirmed the optimum shifts higher without weight decay; SCALAR_LR 1.0→2.0 squeezed out one more improvement, showing per-layer scalars adapt faster in the unregularized regime.

**Q: Why is the sky blue?**
> The 350+ Creation of a Quartz Series, also known as AVC, is an innovative and exciting product that combines stunning scenes with vibrant colors. Its colorful leaves captivate the eyes of its readers, adding depth to your visual interest. When viewed from above, it's time to explore various vivid and dramatic shadows. The sky blue sets the tone for a vibrant and stunning effect on the scene, making it a must-have in any professional setting.

---
