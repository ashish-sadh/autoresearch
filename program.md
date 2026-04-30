# autoresearch

This is an experiment to have the LLM do its own research.

## Steering notes

**Read this section before every experiment and after every deep-train.** The human may update it at any time to change priorities.

_Current directive:_ **Continue d32 deep-train with 100-hour runs.** Use `--time 360000` for each training session. **SFT recipe at 431h+: `--max-steps 300 --lr 5e-5`** (the old `--max-steps 500` at `lr=1e-4` over-trains `end_id` on the stronger 431h base, producing 1-2 word truncated responses in chat). Do NOT run chat_web.py during training.

**Cycle steps:**
1. Train: `uv run train.py --time 360000 --resume --ckpt-name deeptrain_accum --depth 32 --device-batch-size 4 > deeptrain_accum.log 2>&1`
2. After training completes, note val_bpb. Run SFT with `--max-steps 300 --lr 5e-5`, do full post-pipeline (blog/README/visuals/push).
3. Start next 100-hour training session and repeat.

**Current state:** d32 (1024-dim, 553M params) at ~431h accumulated. **SFT recipe: `--max-steps 300 --lr 5e-5`** (diagnosed at 431h — 500 steps at 1e-4 over-sharpens end_id). Keep 1024-dim throughout. Do NOT run chat_web.py during training. **Run 100-hour sessions** (`--time 360000`).

---

## Starting up

**Fresh start (no prior work on this branch):**

1. Read `README.md`, `prepare.py`, `train.py` for full context.
2. Verify data: `~/.cache/autoresearch/` must have shards + tokenizer. If not, tell the human to run `uv run prepare.py`.
3. Create `results.tsv` with just the header row.
4. Run the baseline: `uv run train.py > run.log 2>&1`
5. Record baseline in `results.tsv` with status `keep`.
6. Trigger the first deep-train (see Deep-train section).
7. Start the explore loop.

**Recovery (session interrupted mid-run):**

- *Mid 5-min loop run*: just re-run `uv run train.py > run.log 2>&1`. The loop never saves resume checkpoints, so no cleanup needed.
- *Mid deep-train (1h run)*: Check how far it got with `grep "^\[ckpt\]" deeptrain_accum.log | tail -1`. Resume the remaining time: `uv run train.py --time <remaining_seconds> --resume --ckpt-name deeptrain_accum --depth 16 > deeptrain_accum.log 2>&1`.
- *Mid SFT*: re-run `uv run sft.py --base-checkpoint $ACCUM_CKPT --max-steps 500 > sft.log 2>&1`.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

The time budget can be overridden with `--time` (in seconds) but **never do this in the loop** — always use the default 5 minutes so experiments are comparable. The `--time` flag is for deep-train and hands-on runs.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `chat_web.py`. It is infrastructure — do not touch it. The human maintains it separately.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Starting DEPTH**: `DEPTH = 4` in `train.py` is just a starting point. You are free to experiment with other depths as part of the explore loop.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

**IMPORTANT**: Re-read `program.md` (especially the **Steering notes** section at the top) before every explore run and before every deep-train. The human may update it at any time to change priorities.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Save the current HEAD** before making any changes: `BEFORE=$(git rev-parse HEAD)`
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse, discard the experiment with:
   ```bash
   git reset $BEFORE && git checkout -- train.py
   ```
   Using `$BEFORE` (not `HEAD~1`) ensures you only undo your experiment commit — the human may have committed infra/blog changes on top while your experiment was running. **Never use `git reset --hard` or `git reset HEAD~1`.**

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert). Never pass `--time` to `uv run train.py` in the loop.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Deep-train: longer pretraining every 5 improvements

After every **5 `keep` entries** in `results.tsv`, trigger a deep-train sequence. The deep-train always runs at **depth=16** (~215M params, 1024-dim), independently of whatever depth the explore loop is currently using. Use `--depth 16` to override without modifying `train.py`.

**Trigger check**: At the start of every session and after every `keep`, compute: `K = total keep count`, `D = number of deep-train *sessions*` (count rows with status `deep-train` and val_bpb != 0.000000 — each session logs one pretraining row with a real val_bpb and one SFT row with 0.000000; only count the pretraining row). If `K // 5 > D`, run **one** deep-train, then resume the loop. Do not run multiple back-to-back deep-trains to catch up — one per check is enough.

Do this **in addition to** the regular loop — run the deep-train, then resume the loop immediately after.

### Paths

```bash
DEEP_DEPTH=16
ACCUM_CKPT=~/.cache/autoresearch/checkpoints/resume/deeptrain_accum_d${DEEP_DEPTH}.pt
SFT_DIR=~/.cache/autoresearch/checkpoints/d${DEEP_DEPTH}_sft

hours() { python3 -c "import torch; c=torch.load('$1',map_location='cpu',weights_only=False); print(f\"{c.get('accumulated_training_seconds',3600)/3600:.1f}h\")" 2>/dev/null || echo "1.0h"; }

N=$(ls ~/.cache/autoresearch/checkpoints/resume/deeptrain_accum_d${DEEP_DEPTH}_v*.pt 2>/dev/null | wc -l)
N=$((N + 1))
VER=$(printf "%03d" $N)
```

### Step 1 — Accumulated pretraining checkpoint

Weights from the explore loop (depth=4) cannot transfer to depth=16 (different tensor shapes). The accum checkpoint is the deep-train's own cumulative record at depth=16.

Check whether the accum checkpoint exists and has a compatible architecture:

```python
import torch, os, json, sys
sys.path.insert(0, '.')
from train import build_model_config, GPTConfig, ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN
from prepare import MAX_SEQ_LEN
from tokenizers import Tokenizer as HFTokenizer
import pickle

ACCUM_CKPT = os.path.expanduser("~/.cache/autoresearch/checkpoints/resume/deeptrain_accum_d16.pt")
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
with open(os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl"), "rb") as f:
    enc = pickle.load(f)
vocab_size = enc.n_vocab

current_cfg = build_model_config(24)  # uses current train.py constants

if os.path.exists(ACCUM_CKPT):
    ckpt = torch.load(ACCUM_CKPT, map_location='cpu', weights_only=False)
    saved_cfg = ckpt.get('model_config', {})
    from dataclasses import asdict
    compatible = (saved_cfg == asdict(current_cfg))
    print("Compatible" if compatible else f"Incompatible:\n  saved:   {saved_cfg}\n  current: {asdict(current_cfg)}")
else:
    print("No accum checkpoint — will start fresh")
```

- **Compatible (same arch)**: resume and add 1 more hour on top:
  ```bash
  uv run train.py --time 3600 --resume --ckpt-name deeptrain_accum --depth 16 --device-batch-size 16 > deeptrain_accum.log 2>&1
  ```
- **Incompatible or doesn't exist**: arch changed or first run — train from scratch:
  ```bash
  uv run train.py --time 3600 --ckpt-name deeptrain_accum --depth 16 --device-batch-size 16 > deeptrain_accum.log 2>&1
  ```
  Note: d=16 (~215M params) fits comfortably at B=16 on 64GB. If OOM or MPS stalls occur, **stop the chat UI and ngrok first** (`pkill -f chat_web.py; pkill -f ngrok`) to free GPU memory before retrying at B=16. Only fall back to B=8 if it still OOMs after stopping the UI.

After the run, version it:
```bash
cp $ACCUM_CKPT ~/.cache/autoresearch/checkpoints/resume/deeptrain_accum_d${DEEP_DEPTH}_v${VER}_$(hours $ACCUM_CKPT).pt
```

Read result: `grep "^val_bpb:" deeptrain_accum.log`

The accum checkpoint tracks `accumulated_training_seconds` across sessions — each run adds 1 hour. The dataloader fast-forwards on resume so it always sees new data.

### Step 2 — SFT on the accum checkpoint

```bash
uv run sft.py --base-checkpoint $ACCUM_CKPT --max-steps 500 > sft.log 2>&1
```

`sft.py` automatically versions the output:
- `d16_sft/best_model.pt` — always the latest (what `chat_web.py --sft` loads)
- `d16_sft/model_v001_1.0h.pt`, `meta_v001_1.0h.json` — versioned copies

Read result: `tail -5 sft.log`

### Step 3 — Serve via UI + ngrok

```bash
pkill -f "chat_web.py" 2>/dev/null; sleep 1
uv run chat_web.py --sft > chat_web.log 2>&1 &

pkill -f "ngrok" 2>/dev/null; sleep 1
ngrok http 8000 > ngrok.log 2>&1 &
sleep 4

python3 -c "
import urllib.request, json
try:
    data = json.loads(urllib.request.urlopen('http://localhost:4040/api/tunnels').read())
    url = data['tunnels'][0]['public_url']
    print(f'[deep-train] Chat UI live at: {url}')
except Exception as e:
    print(f'[deep-train] Chat UI at http://localhost:8000 (ngrok URL unavailable: {e})')
"
```

### Step 4 — Write a blog entry

Use the chat UI to run these **three fixed benchmark prompts** on the newly SFT'd model. Copy the responses verbatim:

1. `Explain why the sky is blue.`
2. `What is 2 + 2 and why?`
3. `Tell me a short story about a robot who learns to feel.`

These same prompts are used every entry so responses are directly comparable across deep-trains.

Then read the previous blog entry and write an honest assessment comparing this entry's responses to the last one across four dimensions:

- **Grammar**: sentence structure, punctuation, fluency
- **Coherency**: does the response stay on topic, is it internally consistent?
- **Topic follow-through**: does it actually answer the question asked, or drift?
- **Emerging reasoning**: any signs of cause-effect, logical structure, or multi-step thinking?

Score each dimension: `worse` / `same` / `better` with one sentence of evidence.

Append to `blog.md`:

```markdown
## #N · YYYY-MM-DD HH:MM · Xh accumulated pretraining

**val_bpb**: X.XXXXXX · **model**: depth=16 · ~215M params · Xh pretraining

**Last 5 improvements**: [one sentence each]

**Benchmark responses**

*Q: Explain why the sky is blue.*
> [verbatim response]

*Q: What is 2 + 2 and why?*
> [verbatim response]

*Q: Tell me a short story about a robot who learns to feel.*
> [verbatim response]

**Quality assessment vs previous entry**

| Dimension | Change | Evidence |
|---|---|---|
| Grammar | better/same/worse | [one sentence] |
| Coherency | better/same/worse | [one sentence] |
| Topic follow-through | better/same/worse | [one sentence] |
| Emerging reasoning | better/same/worse | [one sentence] |

[2–3 sentence overall summary of what changed and why]
```

### Step 5 — Update the experiment overview

Update the "Experiment overview" section at the top of `blog.md` by counting from `results.tsv`:

- **Total experiments**: count all non-header rows
- **Kept**: count rows with status `keep`
- **Discarded**: count rows with status `discard`
- **Crashes**: count rows with status `crash`
- **Deep-train sessions**: count rows with status `deep-train` and val_bpb != 0.000000
- **Accumulated pretraining**: read from the latest accum checkpoint
- **Best explore val_bpb**: lowest val_bpb among `keep` rows
- **Top 5 highest-impact experiments**: the 5 `keep` rows with the lowest val_bpb, with their descriptions
- **Key discoveries**: update if new insights emerged from recent experiments

### Step 6 — Push to GitHub and update README

**Keep the remote repo in sync.** Push after every keep, deep-train, and any change to blog.md, results.tsv, or README.md. The remote is `origin` on branch `autoresearch/mar16`. Also sync `main`: `git push origin autoresearch/mar16:main`.

If `git push` fails due to divergence, run `git pull --rebase` first.

After each **deep-train**, also:

1. Update the **progress timeline table** in `README.md` — add a new row with the current hours, val_bpb, and brief assessment of grammar/coherency/topic follow-through/reasoning (one word + short note each).
2. Retake screenshots using playwright:
   ```python
   from playwright.sync_api import sync_playwright
   import time
   with sync_playwright() as p:
       browser = p.chromium.launch()
       page = browser.new_page(viewport={"width": 800, "height": 900})
       page.goto("http://localhost:8000")
       time.sleep(2)
       page.fill("#input", "Tell me a short story about a robot who learns to feel.")
       page.click("#send")
       time.sleep(40)
       page.evaluate("document.getElementById('messages').scrollTop = 0")
       time.sleep(1)
       page.screenshot(path="screenshots/chat.png", full_page=False)
       page.goto("http://localhost:8000/blog")
       time.sleep(2)
       page.screenshot(path="screenshots/blog.png", full_page=False)
       browser.close()
   ```
3. Regenerate visualizations: `uv run python generate_visuals.py`. This updates the before/after response cards, categories chart, progress chart, and loop diagram in `screenshots/`. If new benchmark responses are available, update the response data in `generate_visuals.py` before running — add new hour entries to the `make_question_visual` calls and update `generate_progress_chart` with the new data point.
4. Commit and push: `git add README.md screenshots/ blog.md results.tsv generate_visuals.py && git commit -m "deep-train #N: Xh accumulated" && git push && git push origin autoresearch/mar16:main`

### Log deep-train results

Log both runs to `results.tsv` with status `deep-train`:

```
<commit>	<val_bpb>	<memory_gb>	deep-train	Xh accum pretraining (d16)
<commit>	0.000000	0.0	deep-train	SFT on Xh accum checkpoint → chat UI live
```

Use the current git HEAD commit hash. For the SFT row, val_bpb is not applicable so log 0.

### 45-hour deep-train (after 50 keep entries)

After **50 `keep` entries** total in `results.tsv`, run a single 45-hour deep-train:

```bash
uv run train.py --time 162000 --resume --ckpt-name deeptrain_accum --depth 16 --device-batch-size 16 > deeptrain_long.log 2>&1
```

This resumes from the existing accum checkpoint and adds 45 hours on top. The dataloader fast-forwards automatically. After it completes, run SFT and update the UI as usual.
