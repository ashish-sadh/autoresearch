# Autoresearch — Live Training Log

This is a live log of an autonomous AI research experiment. A Claude agent runs in a loop, modifying a small language model's architecture and training setup in 5-minute experiments on a Mac (Apple M5 Max, 64GB unified memory). Changes that lower validation bits-per-byte (val_bpb — lower is better) are kept; others are discarded.

Every 5 improvements, a deep-train is triggered at depth=16 (~215M parameters, 1024-dim) and trained for 1 hour on web text (climbmix-400b, a 400B token dataset), then fine-tuned for chat using SmolTalk instruction data. Each entry below marks one of those deep-train milestones.

The explore loop runs at a small, fast depth (typically ~5M params, ~500 experiments/day). The deep-train runs at fixed depth=16 (~215M params, 1024-dim, 1 hour each). The larger model trains from scratch or resumes its own accumulated checkpoint — it doesn't transfer weights from the small model, but benefits from hyperparameter insights found in the explore loop.

**Dataset**: climbmix-400b web text (300 shards, ~18B tokens downloaded)
**Hardware**: Apple M5 Max, 64GB unified memory, macOS MPS
**Optimizer**: Muon (matrix params) + AdamW (embeddings/scalars)

### Experiment overview

**Total experiments**: 126 · **Kept**: 34 · **Discarded**: 88 · **Crashes**: 0
**Deep-train sessions**: 2 · **Accumulated pretraining**: 2.0h
**Best explore val_bpb**: 1.295171

**Top 5 highest-impact experiments**

| val_bpb | Description |
|---|---|
| 1.295171 | MATRIX_LR 0.06→0.055 with Muon ns_steps=4 |
| 1.295828 | Muon ns_steps 5→4 — faster steps = more gradient updates |
| 1.301823 | MATRIX_LR 0.04→0.06 — first big LR jump |
| 1.303503 | MATRIX_LR 0.070→0.075 in WD=0.0 regime |
| 1.315839 | lm_head ns_steps 4→5 — tall matrix needs more precise polar decomp |

**Key discoveries**: removing weight decay was a major win for small models; Muon optimizer benefits from tuning ns_steps per matrix shape; learning rates shift higher in the no-WD regime; value embeddings (alternating layers) are critical for quality.

---

*Entries below are added automatically after each deep-train. Newest at the bottom.*

---

## #1 · 2026-03-20 13:45 · 1.0h accumulated pretraining

**val_bpb**: 1.336272 · **model**: depth=16, 1024-dim · ~285M params · 1h pretraining

**Last 5 improvements**: MATRIX_LR 0.065→0.070 continued the trend of higher LR in the WD=0.0 regime; SCALAR_LR 1.0→2.0 showed per-layer scalars adapt faster without regularization; FINAL_LR_FRAC 0.0→0.01 avoided over-decaying learning rate at training end; ve_gate_channels 32→8 sped up steps enough to offset the smaller gate; MATRIX_LR 0.070→0.075 gave the biggest single improvement in recent history.

**Benchmark responses**

*Q: Explain why the sky is blue.*
> The light of darker sunscheds appears to appear as a blue browth blue, which is a delicate blue, green color blue that has been blue, but an red hair gray green green red, where brown. Since bright colors are yellowing, green eyes can be seen on the skin and can be seen in white. But when light changes seem incorrectly close to normal conditions, they may even have blue blue or red. But as darker sunscheds can become green, which is not always known for its blue. As light changes, black skin redness indicates the color of yellow. To avoid this problem, the sky appears white, so it's witnessed that the orange colors are brown or red.

*Q: What is 2 + 2 and why?*
> To solve this problem, we multiply the two terms: 6.3 - 1 = x-ray bolder The answer is: "tool_call>[{"name": \frac{n}}] Then

*Q: Tell me a short story about a robot who learns to feel.*
> The robot is now part of the conversation, and there are several options you can convey or provide. First, let's add some more details for your robot: 1. **Epit**: A robot will be able to control its length, which means that they are designed specifically for a specific number of hours spent. This gives them some time in the process until each one is complete. 2. **Avoid overcrowded or logical**: Consider all that is needed. 3. **Cost For**: A robot will be able to control its length and perform tasks in general, while maintaining an object's positioning speed is essential for keeping the entire body aligned with each other. 4. **Summary**: The robot should transport the distance between a single one and maintains its length, which means that it can be used to track its length.

**Quality assessment** *(baseline — no prior entry to compare)*

| Dimension | Baseline observation |
|---|---|
| Grammar | Sentences are individually grammatical but hallucinated; invented words like "sunscheds" and "browth" appear |
| Coherency | None — responses drift randomly, sky answer wanders through colors/skin/hair with no structure |
| Topic follow-through | Zero — no question is actually answered; math produces pseudo-code/LaTeX fragments; robot story becomes a spec sheet |
| Emerging reasoning | None — numbered lists are cosmetic formatting only, no causal structure or logical steps |

Baseline for the d16 architecture (1024-dim, 285M params) with 1h pretraining. The model has picked up formatting patterns (bold headers, numbered lists, LaTeX-like fragments, even "tool_call" tokens) from the training data, but applies them nonsensically. Quality should improve as accumulated pretraining hours grow.

---

## #2 · 2026-03-20 18:30 · 2.0h accumulated pretraining

**val_bpb**: 1.271953 · **model**: depth=16, 1024-dim · ~285M params · 2h pretraining

**Last 5 improvements**: EMBEDDING_LR 0.4→0.45 gave marginal improvement in the WD=0.0 regime (the only new keep since last deep-train). Prior improvements carried forward: MATRIX_LR 0.070→0.075 was the biggest recent win; ve_gate_channels 32→8 traded gate capacity for more steps; FINAL_LR_FRAC 0.0→0.01 avoided over-decay; SCALAR_LR 1.0→2.0 and MATRIX_LR 0.065→0.070 tuned the no-regularization optimum.

**Benchmark responses**

*Q: Explain why the sky is blue.*
> The sky is tied to an atmosphere of two, so it's a moment in which time. Therefore, it appears that the Sun will be blue is not blue but its geometric nature and longevity!

*Q: What is 2 + 2 and why?*
> To solve the problem, let's start by focusing on the 4 + 6 - 7 - 8 - 5. Consider your starting point of view of the problem list:

*Q: Tell me a short story about a robot who learns to feel.*
> <tool_call>[{"name": "arguments", "arguments": {"motivate self"}, "strongest self-helpness"}]</tool_Cancellations: The robot's "front" is expressed in the human body. It has a dual expression for their own words and life, meaning they have to be more productive or even more effective. The robot needs a differential expressions from each person who must understand what she takes into account:

**Quality assessment vs previous entry**

| Dimension | Change | Evidence |
|---|---|---|
| Grammar | better | No invented words; sentences are shorter and more natural; "sunscheds"/"browth" artifacts gone |
| Coherency | better | Sky response stays focused on sun/sky/atmosphere — short but thematically consistent for the first time |
| Topic follow-through | better | Sky answer mentions "Sun" and "atmosphere" — first time a response touches the actual topic; math attempts "to solve the problem" before drifting |
| Emerging reasoning | same | No real causal structure; math lists numbers without logic; robot emits tool_call fragments |

val_bpb improved significantly (1.272 vs 1.336) with 2x more pretraining. The sky response is notably better — it's short, mentions the sun and atmosphere, and doesn't hallucinate colors endlessly. The math response attempts problem-solving framing ("To solve the problem...") before drifting. The robot story still collapses into tool_call artifacts. Overall: first signs of topic awareness emerging, though no real reasoning yet.

---
