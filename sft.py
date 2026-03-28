"""
Supervised Fine-Tuning (SFT) for the autoresearch base model.

Loads the best pretrained checkpoint and fine-tunes on SmolTalk — a high-quality
instruction dataset — teaching the model to answer questions in a chat format.

The base model learns: "when I see <user>question<end><assistant>, I should produce
a helpful answer followed by <end>."

Usage:
    uv run sft.py                  # fine-tune best d4 checkpoint
    uv run sft.py --depth 8        # for a different depth
    uv run sft.py --max-steps 1000 # shorter run for testing
    uv run sft.py --shards 2       # use more data (slower)

Checkpoint saved to:
    ~/.cache/autoresearch/checkpoints/d{depth}_sft/best_model.pt
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

import requests
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=2000, help="Max optimizer steps")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--shards", type=int, default=1, help="Number of SmolTalk shards to download")
parser.add_argument("--base-checkpoint", type=str, default=None, help="Load base model from a deep-train checkpoint .pt file instead of auto-detecting best_model.pt")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths — auto-detect depth from whichever checkpoint exists
# ---------------------------------------------------------------------------

CACHE_DIR       = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR   = os.path.join(CACHE_DIR, "tokenizer")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")

def find_base_checkpoint():
    """Return (depth, dir) for the largest trained base checkpoint available."""
    if not os.path.exists(CHECKPOINTS_DIR):
        return None, None
    candidates = []
    for name in os.listdir(CHECKPOINTS_DIR):
        if name.startswith("d") and not name.endswith("_sft"):
            model_path = os.path.join(CHECKPOINTS_DIR, name, "best_model.pt")
            if os.path.exists(model_path):
                try:
                    candidates.append((int(name[1:]), os.path.join(CHECKPOINTS_DIR, name)))
                except ValueError:
                    pass
    if not candidates:
        return None, None
    return max(candidates)  # largest depth wins

if args.base_checkpoint:
    detected_depth, BASE_CKPT_DIR = None, None  # will derive from checkpoint
else:
    detected_depth, BASE_CKPT_DIR = find_base_checkpoint()
    if BASE_CKPT_DIR is None:
        print("No base checkpoint found. Run 'uv run train.py' first.")
        sys.exit(1)

if args.base_checkpoint:
    # Derive depth from deep-train checkpoint metadata
    _ckpt_meta = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    _cfg = _ckpt_meta.get("model_config")
    if _cfg is None:
        print("--base-checkpoint file has no model_config — was it saved by train.py with --ckpt-name?")
        sys.exit(1)
    detected_depth = _cfg["n_layer"]
    accum_s = _ckpt_meta.get("accumulated_training_seconds", 0)
    print(f"Loading deep-train checkpoint: depth={detected_depth}, accumulated {accum_s/3600:.1f}h pretraining")

SFT_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, f"d{detected_depth}_sft")
SMOLTALK_DIR    = os.path.join(CACHE_DIR, "smoltalk")
MAX_SEQ_LEN     = 2048

os.makedirs(SFT_CKPT_DIR, exist_ok=True)
os.makedirs(SMOLTALK_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device_type = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Device: {device_type}")

# ---------------------------------------------------------------------------
# Model (mirror of train.py — same architecture)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    window_pattern: str = "L"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 8
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        window = window_size[0]
        if window > 0 and window < T:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().float(), freqs.sin().float()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window, short_window = config.sequence_len, config.sequence_len // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)
        return sizes

    def forward(self, idx, targets=None):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx).float())
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).float() if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        softcap = 15.0
        logits = softcap * torch.tanh(self.lm_head(norm(x)) / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------

tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
if not os.path.exists(tokenizer_pkl):
    print("Tokenizer not found. Run 'uv run prepare.py' first.")
    sys.exit(1)

with open(tokenizer_pkl, "rb") as f:
    enc = pickle.load(f)

bos_id   = enc.encode_single_token("<|reserved_0|>")
user_id  = enc.encode_single_token("<|reserved_1|>")
asst_id  = enc.encode_single_token("<|reserved_2|>")
end_id   = enc.encode_single_token("<|reserved_3|>")
print(f"Tokenizer loaded. Special tokens: bos={bos_id} user={user_id} asst={asst_id} end={end_id}")

# ---------------------------------------------------------------------------
# Load base model checkpoint
# ---------------------------------------------------------------------------

if args.base_checkpoint:
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    meta = {"val_bpb": "n/a (deep-train checkpoint)"}
    print(f"Base model: {config}")
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
else:
    base_model_path = os.path.join(BASE_CKPT_DIR, "best_model.pt")
    base_meta_path  = os.path.join(BASE_CKPT_DIR, "best_meta.json")
    if not os.path.exists(base_model_path):
        print(f"No base checkpoint at {base_model_path}. Run 'uv run train.py' first.")
        sys.exit(1)
    with open(base_meta_path) as f:
        meta = json.load(f)
    config = GPTConfig(**meta["model_config"])
    print(f"Base model: {config}")
    print(f"Base val_bpb: {meta['val_bpb']:.6f}")
    model = GPT(config).to(device)
    state = torch.load(base_model_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)

model.train()
print("Base model loaded.")

# ---------------------------------------------------------------------------
# Download SmolTalk
# ---------------------------------------------------------------------------

def get_smoltalk_urls(num_shards):
    """Fetch parquet file URLs from HuggingFace datasets server API."""
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=HuggingFaceTB/smoltalk"
    print(f"Fetching SmolTalk file list...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    all_files = resp.json()["parquet_files"]
    train_urls = [f["url"] for f in all_files if f["split"] == "train" and f["config"] == "all"]
    return train_urls[:num_shards]


def download_smoltalk(num_shards):
    urls = get_smoltalk_urls(num_shards)
    paths = []
    for i, url in enumerate(urls):
        filename = f"train_{i:05d}.parquet"
        filepath = os.path.join(SMOLTALK_DIR, filename)
        if os.path.exists(filepath):
            print(f"  Shard {i}: already downloaded")
        else:
            print(f"  Shard {i}: downloading from {url[:80]}...")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            tmp = filepath + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            print(f"  Shard {i}: saved to {filepath}")
        paths.append(filepath)
    return paths


print(f"\nDownloading SmolTalk ({args.shards} shard(s))...")
shard_paths = download_smoltalk(args.shards)

# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def format_conversation(messages):
    """
    Tokenize a conversation into (input_ids, loss_mask).
    Loss is only computed on assistant response tokens.

    Format:
      [BOS] [user] Q tokens [end] [asst] A tokens [end] [user] Q2 [end] [asst] A2 [end] ...
    """
    tokens = [bos_id]
    mask   = [0]  # no loss on BOS

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            content_tokens = enc.encode_ordinary(content)
            tokens += [user_id] + content_tokens + [end_id]
            mask   += [0] * (1 + len(content_tokens) + 1)  # no loss on user turns
        elif role == "assistant":
            content_tokens = enc.encode_ordinary(content)
            tokens += [asst_id] + content_tokens + [end_id]
            mask   += [0] + [1] * len(content_tokens) + [1]  # loss on assistant response + end

    return tokens, mask


print("Loading and tokenizing conversations...")
all_examples = []  # list of (tokens, mask)

for path in shard_paths:
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        # SmolTalk has a "messages" column
        col_names = rg.schema.names
        msg_col = "messages" if "messages" in col_names else col_names[0]
        rows = rg.column(msg_col).to_pylist()
        for row in rows:
            if row is None:
                continue
            messages = row if isinstance(row, list) else row.get("messages", [])
            tokens, mask = format_conversation(messages)
            # Skip if no assistant tokens or too short
            if sum(mask) == 0 or len(tokens) < 4:
                continue
            # Truncate to MAX_SEQ_LEN
            tokens = tokens[:MAX_SEQ_LEN + 1]
            mask   = mask[:MAX_SEQ_LEN + 1]
            all_examples.append((tokens, mask))

print(f"Loaded {len(all_examples):,} conversations.")
if len(all_examples) == 0:
    print("No usable conversations found. Check SmolTalk format.")
    sys.exit(1)

random.seed(42)
random.shuffle(all_examples)

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_batch(examples, batch_size):
    """Yield batches of (input_ids, targets) tensors with loss masking."""
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        max_len = max(len(t) for t, m in batch)

        input_ids = torch.zeros(len(batch), max_len - 1, dtype=torch.long)
        targets   = torch.full((len(batch), max_len - 1), -1, dtype=torch.long)

        for j, (tokens, mask) in enumerate(batch):
            n = len(tokens) - 1
            input_ids[j, :n] = torch.tensor(tokens[:-1], dtype=torch.long)
            for k in range(n):
                if mask[k + 1]:  # compute loss on next token if it's in assistant response
                    targets[j, k] = tokens[k + 1]

        yield input_ids.to(device), targets.to(device)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

total_steps = min(args.max_steps, len(all_examples) // args.batch_size)
print(f"\nTraining for {total_steps} steps (batch_size={args.batch_size}, lr={args.lr})")

# Cosine LR decay
def get_lr(step):
    return args.lr * 0.5 * (1 + torch.cos(torch.tensor(step / total_steps * 3.14159)).item())

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
t0 = time.time()
step = 0
best_loss = float("inf")
smooth_loss = None

for epoch in range(999):
    for input_ids, targets in make_batch(all_examples, args.batch_size):
        if step >= total_steps:
            break

        # LR schedule
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item()
        smooth_loss = loss_val if smooth_loss is None else 0.95 * smooth_loss + 0.05 * loss_val
        dt = time.time() - t0
        remaining = (total_steps - step) * (dt / max(step, 1))

        print(f"\rstep {step+1:04d}/{total_steps} | loss: {smooth_loss:.4f} | lr: {lr:.2e} | elapsed: {dt:.0f}s | remaining: {remaining:.0f}s    ",
              end="", flush=True)
        step += 1

        # Save intermediate checkpoint every 100 steps
        if step % 100 == 0:
            _inter_state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model.state_dict().items()}
            _inter_path = os.path.join(SFT_CKPT_DIR, "best_model.pt")
            torch.save(_inter_state, _inter_path)
            print(f"\n[ckpt] saved intermediate SFT at step {step}", flush=True)

    if step >= total_steps:
        break

print("\nTraining complete.")

# ---------------------------------------------------------------------------
# Save SFT checkpoint
# ---------------------------------------------------------------------------

model.eval()
model_state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model.state_dict().items()}

# Determine accumulated pretraining hours from the base checkpoint
_accum_hours = 0.0
if args.base_checkpoint:
    _accum_hours = _ckpt_meta.get("accumulated_training_seconds", 0) / 3600

# Auto-version: count existing versioned model files to pick next version number
_existing_versions = [f for f in os.listdir(SFT_CKPT_DIR) if f.startswith("model_v") and f.endswith(".pt")]
_ver_n = len(_existing_versions) + 1
_ver_str = f"v{_ver_n:03d}"
_hours_str = f"{_accum_hours:.1f}h"

model_path = os.path.join(SFT_CKPT_DIR, "best_model.pt")
torch.save(model_state, model_path)
versioned_model_path = os.path.join(SFT_CKPT_DIR, f"model_{_ver_str}_{_hours_str}.pt")
torch.save(model_state, versioned_model_path)

sft_meta = {
    "step": step,
    "final_loss": smooth_loss,
    "base_val_bpb": meta["val_bpb"],
    "model_config": asdict(config),
    "vocab_size": enc.n_vocab,
    "sft_steps": step,
    "accum_hours": _accum_hours,
    "version": _ver_str,
}
meta_path = os.path.join(SFT_CKPT_DIR, "best_meta.json")
with open(meta_path, "w") as f:
    json.dump(sft_meta, f, indent=2)
versioned_meta_path = os.path.join(SFT_CKPT_DIR, f"meta_{_ver_str}_{_hours_str}.json")
with open(versioned_meta_path, "w") as f:
    json.dump(sft_meta, f, indent=2)

print(f"\nSaved SFT model → {model_path}")
print(f"Versioned copy  → {versioned_model_path}")
print(f"Saved SFT meta  → {meta_path}")
print(f"\nRun the chat UI with:")
print(f"  uv run chat_web.py --sft")
