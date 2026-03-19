# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

The time budget can be overridden with `--time` (in seconds) but **never do this in the loop** — always use the default 5 minutes so experiments are comparable. The `--time` flag is for the human to do hands-on or overnight runs after the loop finishes.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

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

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert). Never pass `--time` to `uv run train.py` in the loop.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Deep-train: longer pretraining every 5 improvements

After every **5 `keep` entries** in `results.tsv`, trigger a deep-train sequence. This takes the current best model and trains it for 1 full hour, producing two checkpoints. Do this **in addition to** the regular loop — do not pause normal experiments for it; run the deep-train, then resume the loop immediately after.

### Paths (resolve DEPTH from `grep "^DEPTH" train.py`)

```
BEST_MODEL=~/.cache/autoresearch/checkpoints/d{DEPTH}/best_model.pt
RESUME_DIR=~/.cache/autoresearch/checkpoints/resume
SFT_DIR=~/.cache/autoresearch/checkpoints/d{DEPTH}_sft
FRESH_CKPT=$RESUME_DIR/deeptrain_fresh_d{DEPTH}.pt
ACCUM_CKPT=$RESUME_DIR/deeptrain_accum_d{DEPTH}.pt
```

Determine the trigger index (1-based count of how many deep-trains have run so far for this depth):

```bash
N=$(ls $RESUME_DIR/deeptrain_accum_d{DEPTH}_v*.pt 2>/dev/null | wc -l)
N=$((N + 1))
VER=$(printf "%03d" $N)   # e.g. 001, 002, 003...

# Helper to extract accumulated hours from a checkpoint
hours() { python3 -c "import torch; c=torch.load('$1',map_location='cpu',weights_only=False); print(f\"{c.get('accumulated_training_seconds',3600)/3600:.1f}h\")" 2>/dev/null || echo "1.0h"; }
```

### Step 1 — Fresh checkpoint (always)

Start from the loop's current best model weights with a fresh optimizer. This always reflects the latest architecture:

```bash
uv run train.py --time 3600 --init-from $BEST_MODEL --ckpt-name deeptrain_fresh > deeptrain_fresh.log 2>&1
# Version it — include accumulated hours in filename
cp $FRESH_CKPT $RESUME_DIR/deeptrain_fresh_d{DEPTH}_v$VER_$(hours $FRESH_CKPT).pt
```

Read result: `grep "^val_bpb:" deeptrain_fresh.log`

### Step 2 — Accumulated checkpoint (cumulative pretraining)

Check whether the accum checkpoint exists and has a compatible architecture:

```python
import torch, os
ACCUM_CKPT = os.path.expanduser("~/.cache/autoresearch/checkpoints/resume/deeptrain_accum_d{DEPTH}.pt")
if os.path.exists(ACCUM_CKPT):
    ckpt = torch.load(ACCUM_CKPT, map_location='cpu', weights_only=False)
    saved_cfg = ckpt.get('model_config', {})
    # compare saved_cfg with current GPTConfig built from train.py constants
    # compatible if all fields match
```

- **Compatible (same arch)**: resume and add 1 more hour on top:
  ```bash
  uv run train.py --time 3600 --resume --ckpt-name deeptrain_accum > deeptrain_accum.log 2>&1
  ```
- **Incompatible or doesn't exist**: arch changed, train from scratch for 1 hour (accum resets for new arch):
  ```bash
  uv run train.py --time 3600 --ckpt-name deeptrain_accum > deeptrain_accum.log 2>&1
  ```

After the run, version it:
```bash
cp $ACCUM_CKPT $RESUME_DIR/deeptrain_accum_d{DEPTH}_v$VER_$(hours $ACCUM_CKPT).pt
```

Read result: `grep "^val_bpb:" deeptrain_accum.log`

The accum checkpoint tracks `accumulated_training_seconds` across sessions — each run adds 1 hour. Over time this gives you 1h, 2h, 3h, ... of continuous pretraining on the best architecture found so far.

### Step 3 — SFT on the accum checkpoint

The accum checkpoint is a base (completion) model. Run SFT to make it chat-capable before serving:

```bash
uv run sft.py --base-checkpoint $ACCUM_CKPT > sft.log 2>&1
# Version the SFT output — inherit hours from the accum it was built on
ACCUM_HOURS=$(hours $ACCUM_CKPT)
cp $SFT_DIR/best_model.pt $SFT_DIR/model_v${VER}_${ACCUM_HOURS}.pt
cp $SFT_DIR/best_meta.json $SFT_DIR/meta_v${VER}_${ACCUM_HOURS}.json
```

This fine-tunes on SmolTalk instruction data and saves the result to `d{DEPTH}_sft/best_model.pt` — the same path that `chat_web.py --sft` auto-detects.

Read result: `tail -5 sft.log`

### Step 4 — Serve via UI + ngrok

After SFT completes, update the live UI:

```bash
# Kill any existing server
pkill -f "chat_web.py" 2>/dev/null; sleep 1

# Start the chat UI in the background (--sft loads the freshly SFT'd deep-train model)
uv run chat_web.py --sft > chat_web.log 2>&1 &

# Start ngrok tunnel (requires ngrok CLI to be installed)
pkill -f "ngrok" 2>/dev/null; sleep 1
ngrok http 8000 > ngrok.log 2>&1 &
sleep 4

# Print the public URL
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

### Log deep-train results

Log all three runs to `results.tsv` with status `deep-train`:

```
<commit>	<val_bpb>	<memory_gb>	deep-train	1h fresh from loop best (d{DEPTH})
<commit>	<val_bpb>	<memory_gb>	deep-train	{N}h accum pretraining (d{DEPTH})
<commit>	0.000000	0.0	deep-train	SFT on {N}h accum checkpoint → chat UI live
```

Use the current git HEAD commit hash (the deep-train doesn't modify `train.py`, so no new commit is needed). For the SFT row, val_bpb is not applicable so log 0.
