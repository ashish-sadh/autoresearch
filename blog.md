# Autoresearch — Live Training Log

This is a live log of an autonomous AI research experiment. A Claude agent runs in a loop, modifying a small language model's architecture and training setup in 5-minute experiments on a Mac (Apple M5 Max, 64GB unified memory). Changes that lower validation bits-per-byte (val_bpb — lower is better) are kept; others are discarded.

Every 5 improvements, a deep-train is triggered at depth=16 (~215M parameters, 1024-dim) and trained for 1 hour on web text (climbmix-400b, a 400B token dataset), then fine-tuned for chat using SmolTalk instruction data. Each entry below marks one of those deep-train milestones.

The explore loop runs at a small, fast depth (typically ~5M params, ~500 experiments/day). The deep-train runs at fixed depth=16 (~215M params, 1024-dim, 1 hour each). The larger model trains from scratch or resumes its own accumulated checkpoint — it doesn't transfer weights from the small model, but benefits from hyperparameter insights found in the explore loop.

**Dataset**: climbmix-400b web text (300 shards, ~18B tokens downloaded)
**Hardware**: Apple M5 Max, 64GB unified memory, macOS MPS
**Optimizer**: Muon (matrix params) + AdamW (embeddings/scalars)

### Experiment overview

**Total experiments**: 148 · **Kept**: 37 · **Discarded**: 103 · **Crashes**: 0
**Deep-train sessions**: 4 · **Accumulated pretraining**: 4.0h
**Best explore val_bpb**: 1.295480

**Top 5 highest-impact experiments**

| val_bpb | Description |
|---|---|
| 1.295171 | MATRIX_LR 0.06→0.055 with Muon ns_steps=4 |
| 1.295480 | MATRIX_LR 0.075→0.080 with new warmup/FINAL_LR settings |
| 1.295828 | Muon ns_steps 5→4 — faster steps = more gradient updates |
| 1.297493 | Muon momentum warmup 200→150 steps |
| 1.298525 | FINAL_LR_FRAC 0.01→0.02 — higher min LR at end |

**Key discoveries**: removing weight decay was a major win for small models; Muon optimizer benefits from tuning ns_steps per matrix shape; learning rates shift higher in the no-WD regime; value embeddings (alternating layers) are critical for quality; schedule parameters (FINAL_LR_FRAC, momentum warmup) have cascading effects — tuning one unlocks better optima for others (e.g. higher FINAL_LR_FRAC enabled higher MATRIX_LR and faster momentum warmup).

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

## #3 · 2026-03-20 23:30 · 3.0h accumulated pretraining

**val_bpb**: 1.214576 · **model**: depth=16, 1024-dim · ~285M params · 3h pretraining

**Last 5 improvements**: EMBEDDING_LR 0.4→0.45 was the only new keep since last deep-train. Prior wins carried forward: MATRIX_LR 0.070→0.075 in WD=0.0 regime; ve_gate_channels 32→8 traded capacity for speed; FINAL_LR_FRAC 0.0→0.01 prevented over-decay; SCALAR_LR 1.0→2.0 accelerated skip-connection adaptation.

**Benchmark responses**

*Q: Explain why the sky is blue.*
> The answer to this question can help you understand how your sky might have been blue. If we know that our moon's stars are blue, then perhaps there may be more stars than ever before. In fact, what if one of these solar panels could lead to an overall atmosphere in astronauts? Especially those reflecting the mysterious earthquakes of Japan's Ontario Union, Yale or Sori might have been a viable alternative for today's planets. This may be true if you are looking at some of these solar panels. Now let's delve into that question – What other solar cells could we now focus on? Let's dive right in! Firstly, what do sky light reflections mean by the moon? By reading the images and exploring how stars can influence sunlight. Happy scrapbook!

*Q: What is 2 + 2 and why?*
> To maintain this, it's a 1) 4 + 0 = -5 (Eldogy), which follows the "length" of the first (i.e., 3.5). However, to avoid this with 2 + 6 + 7 + 9, we use a 4 + 8 formula for adding 1) to achieve the maximum value and 4 = -25. To create an equally 0; In addition, you can add (a *{"5*} / * "t", "length=\/c"...) The equation given for each = is: 1 + = (e.g., 2)

*Q: Tell me a short story about a robot who learns to feel.*
> It's true that it doesn't necessarily have any or all of your feet in the field. But there is such an option as "relying off" a way to get your hands straight across one. Lastly, if you've managed to achieve more goal with each object level and then want them smoothly, make sure they'll start quickly.

**Quality assessment vs previous entry**

| Dimension | Change | Evidence |
|---|---|---|
| Grammar | better | Sentences are longer and more fluent; no invented words; contractions and punctuation used naturally |
| Coherency | same | Sky answer still drifts (solar panels, earthquakes, scrapbook); math produces pseudo-formulas; robot story is off-topic |
| Topic follow-through | same | Sky mentions "sky" and "blue" and "atmosphere" but doesn't explain causation; math mentions "4" but reasoning is wrong; robot story doesn't mention robots |
| Emerging reasoning | same | Question-answer framing improved ("The answer to this question..."), rhetorical structure emerging ("Let's dive right in!"), but no actual causal reasoning |

val_bpb improved markedly (1.215 vs 1.272, a 4.5% drop). Grammar is noticeably better — sentences are longer, more natural, and free of invented words. The model has picked up rhetorical patterns ("Let's delve into...", "Let's dive right in!") and question-framing structures. However, coherency and topic follow-through remain poor — the sky answer wanders into solar panels and earthquakes, and the robot story doesn't mention robots at all. The model is learning to write fluently but not yet to stay on topic or reason.

---

## #4 · 2026-03-21 01:00 · 4.0h accumulated pretraining

**val_bpb**: 1.137062 · **model**: depth=16, 1024-dim · ~285M params · 4h pretraining

**Last 5 improvements**: FINAL_LR_FRAC 0.01→0.02 showed a higher minimum LR at end of warmdown helps; Muon momentum warmup 200→150 steps found faster ramp works better for d4; MATRIX_LR 0.075→0.080 was unlocked by the new schedule settings. These three cascading improvements demonstrate that schedule parameters interact — tuning one opens new optima for others.

**Benchmark responses**

*Q: Explain why the sky is blue.*
> Here are some of the most popular photos in this series: 1) Ahhh, Haha! Then you can search for something from your favorite store (which might be a great idea), we've got a little bit more or less information about it. 2) Banding around the sky is not just about gathering data to share knowledge. It's also about making sense of what makes our thoughts so interesting and informative. Here are some suggestions that you can start with: 1) What colors will I mix? 2) Ahh, Haha! It's a great idea to add details such as the glow from your favorite store, rather than color changes. 3) Blacker Thrives: Superbugs are often less interesting and might give you more clarity or anchor over their colors.

*Q: What is 2 + 2 and why?*
> SearchTextCreate a link. To find out what happens when you want to use the product, please enter on your newcomb. The product (couldn't be used), was an object with no point at all. This article discusses some of its uses: Firstly, we can create something or describe it using our own way. Here are some properties that affect how many devices run under 2: Multiple-earned. Sets tend not have their own strengths at all and the differences between them.

*Q: Tell me a short story about a robot who learns to feel.*
> The robots are an essential part of the daily life cycle, so they need regular checkups and maintenance at their peak. Here's how you can make them easier: this will help keep track of when to take care of your young. First things first, let me tell you about a robot who wants it to do something new. Firstly, you must regularly clean up any debris and dust that might be causing the issue. This could mean cleaning the area with debris or even removing dirt from other objects. Secondly, if your child feels irritable or loses interest in their young, they may need immediate check-ups to prevent potential problems.

**Quality assessment vs previous entry**

| Dimension | Change | Evidence |
|---|---|---|
| Grammar | better | More complex sentence structures; natural use of colons, parentheticals, and list formatting; no invented words |
| Coherency | same | Sky answer still drifts (stores, photos, superbugs); math produces product/link jargon; robot mixes childcare with maintenance |
| Topic follow-through | better | Robot answer mentions "robots" and "checkups/maintenance" — first time the robot story relates to the topic; sky mentions "colors" and "sky" |
| Emerging reasoning | better | Robot story uses temporal ordering ("First things first", "Firstly", "Secondly"); sky lists numbered suggestions; structure is becoming more organized |

val_bpb dropped significantly (1.137 vs 1.215, a 6.4% improvement). The model continues to improve in grammar and structure — sentences are more complex and naturally punctuated. The robot story is notably better: it mentions robots, discusses maintenance and checkups, and uses sequential reasoning ("First... Secondly..."). However, the content still doesn't truly answer the questions. The sky answer is about colors/stores rather than Rayleigh scattering, and the math answer generates product-description jargon instead of arithmetic. The model is learning to organize thoughts but not yet to produce factual content.

---
