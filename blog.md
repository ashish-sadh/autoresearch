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

**Benchmark responses**

*Q: Explain why the sky is blue.*
> The sky's white and greenish color, while it is darkness. Slowly flowing water into one of two different colors: a deep red (Green), grey or orange. We can see that in the picture above, "melting" is not just about the shadows. This is an evident, but mystical and infamously infamous idea. I am glad we have been using our observations for some time now with the sky's white to blue itself! Here are a few common ways you can create this visual experience: 1. Call "Power": Green or green lines, such as Grey or dark color, often come from other colors like red or yellow. 2. Flowering water: A rainbow of blue and gray can be found in a very vibrant mix, so it is best to try the same size again.

*Q: What is 2 + 2 and why?*
> The answer to this question can depend on several factors. First, the type of fluid in a system should be considered as "cool" or "when an entire system is connected," because it's more typically used for cooling purposes. To determine if you need extra coolant at home and would like to know what specifications are required or how often you're going to install them, the answer will depend on various factors. 1. Country: The type of fluid in a system should be considered as "cool" or "when an entire system is connected." This means your installation may not always perform like calculating specifications.

*Q: Tell me a short story about a robot who learns to feel.*
> We have the world's most popular inhabitants, and we know that you can expect a few of these ideas to be quite powerful. This is a great opportunity for learners to explore all aspects of your life and get started with this incredible activity. Here are some interesting stories from our perspective on robotics: 1. Error: This book delves into the world of robots that can provide you with opportunities to engage in activities like reading, playing sports, or engaging in physical activity. 2. Tell me about a new robot with an idea worth it and how they develop their own sense of intellect and motivation.

**Quality assessment** *(baseline — no prior entry to compare)*

| Dimension | Baseline observation |
|---|---|
| Grammar | Sentences are individually grammatical but content is pure hallucination |
| Coherency | None — each sentence contradicts or ignores the previous |
| Topic follow-through | Zero — all three questions completely ignored; model outputs pseudo-listicle filler |
| Emerging reasoning | None — no causal structure, no logical steps, no self-reference to the question |

This is the 1h baseline on a small 256-dim 24-layer architecture (~52M params). The model has absorbed enough web-text patterns to produce fluent-looking English with formatting (numbered lists, bold), but has no grounding in meaning. All future entries use the full 1536-dim ~860M param model with progressively more pretraining hours.

---
