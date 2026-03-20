"""
Web UI for the autoresearch model.

Two modes:
  Base model (default) — text completion, type a prompt and the model continues it.
  SFT model (--sft)    — chat mode, ask questions and get answers.

Usage:
    uv run chat_web.py            # base model text completion
    uv run chat_web.py --sft      # chat with SFT fine-tuned model
    uv run chat_web.py --depth 8
    uv run chat_web.py --port 8080

Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--sft", action="store_true", help="Load SFT fine-tuned model and use chat mode")
parser.add_argument("--checkpoint", type=str, default=None, help="Load directly from a resume checkpoint .pt file (e.g. deeptrain_accum_d8.pt)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model (mirror of train.py — inference only, no optimizer)
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
    assert x.ndim == 4
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
        self.ve_gate_channels = 32
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

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx).float())
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).float() if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        softcap = 15.0
        logits = self.lm_head(norm(x))
        return softcap * torch.tanh(logits / softcap)

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

def find_checkpoint(sft=False):
    """Auto-detect the best available checkpoint (largest depth)."""
    if not os.path.exists(CHECKPOINTS_DIR):
        return None
    suffix = "_sft" if sft else ""
    candidates = []
    for name in os.listdir(CHECKPOINTS_DIR):
        if name.startswith("d") and name.endswith(suffix) and (not sft or name.endswith("_sft")):
            model_path = os.path.join(CHECKPOINTS_DIR, name, "best_model.pt")
            if os.path.exists(model_path):
                try:
                    depth = int(name[1:].replace("_sft", ""))
                    candidates.append((depth, os.path.join(CHECKPOINTS_DIR, name)))
                except ValueError:
                    pass
    if not candidates:
        return None
    return max(candidates)[1]  # largest depth

device_type = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Device: {device_type}")

CHECKPOINT_DIR = None  # set below if not using --checkpoint

if args.checkpoint:
    # Load from a resume/deep-train checkpoint directly
    print(f"Loading deep-train checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    config = GPTConfig(**cfg)
    print(f"Model config: {config}")
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    accum_s = ckpt.get("accumulated_training_seconds", 0)
    print(f"Loaded deep-train model (accumulated {accum_s/3600:.1f}h of pretraining, step {ckpt.get('step','?')})")
    mode_label = "deep-train base completion"
else:
    CHECKPOINT_DIR = find_checkpoint(sft=args.sft)
    if CHECKPOINT_DIR is None:
        msg = "No SFT checkpoint found. Run 'uv run sft.py' first." if args.sft else "No checkpoint found. Run 'uv run train.py' first."
        print(msg)
        sys.exit(1)
    print(f"Loading checkpoint from {CHECKPOINT_DIR}...")
    meta_path = os.path.join(CHECKPOINT_DIR, "best_meta.json")
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    with open(meta_path) as f:
        meta = json.load(f)
    cfg = meta["model_config"]
    config = GPTConfig(**cfg)
    print(f"Model config: {config}")
    model = GPT(config).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    mode_label = "SFT chat" if args.sft else "base completion"
    val_info = f"val_bpb={meta.get('val_bpb', meta.get('base_val_bpb', '?')):.6f}" if not args.sft else f"sft_steps={meta.get('sft_steps','?')}"
    print(f"Loaded model ({mode_label}, {val_info})")

tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
with open(tokenizer_pkl, "rb") as f:
    enc = pickle.load(f)
bos_id  = enc.encode_single_token("<|reserved_0|>")
user_id = enc.encode_single_token("<|reserved_1|>")
asst_id = enc.encode_single_token("<|reserved_2|>")
end_id  = enc.encode_single_token("<|reserved_3|>")
print(f"Tokenizer loaded (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Model stats footer
# ---------------------------------------------------------------------------

_num_params = sum(p.numel() for p in model.parameters())
_accum_hours = 0.0
_tokens_seen = 0

if args.checkpoint:
    _accum_hours = ckpt.get("accumulated_training_seconds", 0) / 3600
    _tokens_seen = ckpt.get("step", 0) * 32768  # approximate
elif args.sft and CHECKPOINT_DIR is not None:
    # Read accum_hours from versioned SFT meta (has accurate accumulated pretraining)
    _versioned_metas = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("meta_v") and f.endswith(".json")])
    if _versioned_metas:
        with open(os.path.join(CHECKPOINT_DIR, _versioned_metas[-1])) as _f:
            _vm = json.load(_f)
        _accum_hours = _vm.get("accum_hours", 0.0)
    # Total tokens seen: read from the base (non-sft) checkpoint meta
    _base_depth = config.n_layer
    _base_dir = os.path.join(CHECKPOINTS_DIR, f"d{_base_depth}")
    _base_meta_path = os.path.join(_base_dir, "best_meta.json")
    if os.path.exists(_base_meta_path):
        with open(_base_meta_path) as _f:
            _base_meta = json.load(_f)
        _tokens_seen = _base_meta.get("total_tokens", 0)

_footer_html = (
    f'<div id="minfo">'
    f'depth {config.n_layer} · {_num_params/1e6:.1f}M params'
    f' · {_tokens_seen/1e9:.2f}B tokens · {_accum_hours:.1f}h pretraining'
    f' · <a href="/blog" style="color:#4a9eff;text-decoration:none">training log →</a>'
    f' · powered by M5 Max · 64GB'
    f'</div>'
)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SPECIAL_IDS = {bos_id, user_id, asst_id, end_id}

@torch.no_grad()
def sample(prompt_tokens: list[int], max_tokens: int, temperature: float, top_k: int,
           stop_token: int = None, repetition_penalty: float = 1.3):
    tokens = list(prompt_tokens)
    for _ in range(max_tokens):
        ctx = tokens[-config.sequence_len:]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(idx)[0, -1]

        # Suppress all special tokens from being generated
        for sid in SPECIAL_IDS:
            logits[sid] = float("-inf")
        # Re-allow the stop token so the model can end its response
        if stop_token is not None:
            logits[stop_token] = model(idx)[0, -1][stop_token]

        # Repetition penalty — discourages repeating recent tokens
        if repetition_penalty > 1.0:
            for tid in set(tokens[-64:]):
                if tid not in SPECIAL_IDS:
                    logits[tid] = logits[tid] / repetition_penalty if logits[tid] > 0 else logits[tid] * repetition_penalty

        if temperature == 0.0:
            next_token = logits.argmax().item()
        else:
            logits = logits / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[-1]] = float("-inf")
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        if next_token == stop_token or next_token == bos_id:
            break
        if next_token in SPECIAL_IDS:
            break
        tokens.append(next_token)
        yield next_token

# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

app = FastAPI()

BASE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch · text completion</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #0d0d0d; color: #e0e0e0; padding: 24px; max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.1rem; color: #aaa; margin-bottom: 4px; }
  p.sub { font-size: 0.8rem; color: #555; margin-bottom: 20px; }
  textarea { width: 100%; height: 120px; background: #1a1a1a; color: #e0e0e0;
            border: 1px solid #333; padding: 12px; font-family: monospace;
            font-size: 0.9rem; resize: vertical; border-radius: 4px; }
  #controls { display: flex; gap: 12px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  label { font-size: 0.8rem; color: #888; }
  input[type=range] { width: 100px; }
  span.val { font-size: 0.8rem; color: #ccc; min-width: 30px; display: inline-block; }
  button { background: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
           padding: 8px 20px; cursor: pointer; border-radius: 4px; font-family: monospace; }
  button:hover { background: #3a3a3a; }
  button:disabled { opacity: 0.4; cursor: default; }
  #output { margin-top: 16px; background: #1a1a1a; border: 1px solid #333;
            padding: 16px; min-height: 200px; white-space: pre-wrap;
            font-size: 0.9rem; line-height: 1.6; border-radius: 4px; }
  .prompt-text { color: #7ec8e3; }
  .gen-text { color: #e0e0e0; }
  #status { font-size: 0.75rem; color: #555; margin-top: 6px; }
  #minfo { font-size: 0.72rem; color: #444; margin-top: 24px; padding-top: 8px; border-top: 1px solid #1e1e1e; }
  #minfo a { color: #555; text-decoration: none; }
  #minfo a:hover { color: #888; }
</style>
</head>
<body>
<h1>autoresearch · base model · text completion</h1>
<p class="sub">Type a prompt. The model continues it. (Base model — not instruction-tuned. Try: "Once upon a time" or "The history of") · <a href="/blog" style="color:#555;text-decoration:none">training log →</a></p>
<textarea id="prompt" placeholder="Once upon a time"></textarea>
<div id="controls">
  <label>temperature <input type="range" id="temp" min="0" max="2" step="0.05" value="0.8"><span class="val" id="temp-val">0.8</span></label>
  <label>top-k <input type="range" id="topk" min="0" max="200" step="5" value="50"><span class="val" id="topk-val">50</span></label>
  <label>max tokens <input type="range" id="maxtok" min="16" max="512" step="16" value="256"><span class="val" id="maxtok-val">256</span></label>
  <button id="btn" onclick="run()">Generate</button>
  <button onclick="stopGen()" id="stop-btn" disabled>Stop</button>
</div>
<div id="output"><span class="prompt-text"></span><span class="gen-text"></span></div>
<div id="status"></div>
<script>
let es = null;
document.querySelectorAll('input[type=range]').forEach(el => {
  el.addEventListener('input', () => document.getElementById(el.id+'-val').textContent = el.value);
});
function stopGen() {
  if (es) { es.close(); es = null; }
  document.getElementById('btn').disabled = false;
  document.getElementById('stop-btn').disabled = true;
  document.getElementById('status').textContent = 'Stopped.';
}
function run() {
  const prompt = document.getElementById('prompt').value;
  if (!prompt.trim()) return;
  document.querySelector('.prompt-text').textContent = prompt;
  document.querySelector('.gen-text').textContent = '';
  document.getElementById('status').textContent = 'Generating...';
  document.getElementById('btn').disabled = true;
  document.getElementById('stop-btn').disabled = false;
  const p = new URLSearchParams({prompt,
    temperature: document.getElementById('temp').value,
    top_k: document.getElementById('topk').value,
    max_tokens: document.getElementById('maxtok').value});
  es = new EventSource('/generate?' + p);
  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.done) { stopGen(); document.getElementById('status').textContent = 'Done.'; return; }
    document.querySelector('.gen-text').textContent += d.token;
  };
  es.onerror = () => { stopGen(); document.getElementById('status').textContent = 'Error.'; };
}
document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run();
});
</script>
</body>
</html>"""
BASE_HTML = BASE_HTML.replace("</body>", _footer_html + "\n</body>")

CHAT_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch · chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #09090b; --surface: #18181b; --border: #27272a; --border-focus: #3b82f6;
    --text: #fafafa; --text-secondary: #a1a1aa; --text-faint: #52525b;
    --blue: #3b82f6; --blue-hover: #2563eb;
    --user-bg: #1e3a5f; --user-text: #dbeafe;
    --asst-bg: #18181b; --asst-border: #27272a;
  }
  html, body { height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column;
    max-width: 720px; margin: 0 auto;
    height: 100dvh;
  }

  /* Header — minimal */
  #header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  #header .title { font-size: 0.85rem; font-weight: 600; color: var(--text); letter-spacing: -0.01em; }
  #header .sub { font-size: 0.7rem; color: var(--text-faint); margin-top: 1px; }
  #header a { font-size: 0.7rem; color: var(--text-faint); text-decoration: none; }
  #header a:hover { color: var(--text-secondary); }

  /* Messages — the hero */
  #messages {
    flex: 1; overflow-y: auto; padding: 20px 16px;
    display: flex; flex-direction: column; gap: 14px;
    -webkit-overflow-scrolling: touch;
  }
  .msg {
    max-width: 80%; padding: 12px 16px;
    font-size: 0.95rem; line-height: 1.65; white-space: pre-wrap; word-break: break-word;
  }
  .user {
    align-self: flex-end; background: var(--user-bg); color: var(--user-text);
    border-radius: 18px 18px 4px 18px;
  }
  .assistant {
    align-self: flex-start; background: var(--asst-bg); border: 1px solid var(--asst-border);
    color: var(--text); border-radius: 4px 18px 18px 18px;
  }
  .assistant.thinking { color: var(--text-faint); font-style: italic; }

  /* Empty state */
  #empty {
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: var(--text-faint); font-size: 0.85rem; text-align: center; padding: 20px;
  }

  /* Input area — prominent */
  #input-area {
    padding: 12px 14px 8px; flex-shrink: 0;
    border-top: 1px solid var(--border);
  }
  #input-row { display: flex; gap: 8px; align-items: flex-end; }
  #input {
    flex: 1; background: var(--surface); color: var(--text);
    border: 1px solid var(--border); padding: 12px 16px;
    border-radius: 24px; font-family: inherit; font-size: 1rem; line-height: 1.45;
    resize: none; min-height: 48px; max-height: 140px; overflow-y: auto;
    outline: none; transition: border-color 0.2s; -webkit-appearance: none;
  }
  #input:focus { border-color: var(--border-focus); box-shadow: 0 0 0 1px var(--border-focus); }
  #input::placeholder { color: var(--text-faint); }
  #send {
    background: var(--blue); color: #fff; border: none;
    width: 48px; height: 48px; border-radius: 50%;
    cursor: pointer; font-size: 1.2rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    transition: background 0.15s, transform 0.1s; -webkit-tap-highlight-color: transparent;
  }
  #send:hover { background: var(--blue-hover); }
  #send:active { transform: scale(0.93); }
  #send:disabled { opacity: 0.2; cursor: default; transform: none; }

  /* Controls — tucked away */
  #controls-toggle {
    display: flex; align-items: center; gap: 4px;
    padding: 6px 0 4px; cursor: pointer; user-select: none;
    font-size: 0.65rem; color: var(--text-faint); border: none; background: none;
  }
  #controls-toggle:hover { color: var(--text-secondary); }
  #controls-toggle .arrow { font-size: 0.5rem; transition: transform 0.2s; }
  #controls-toggle.open .arrow { transform: rotate(90deg); }
  #controls {
    display: none; gap: 8px 16px; padding: 6px 0 2px; flex-wrap: wrap;
  }
  #controls.open { display: flex; }
  #controls label {
    font-size: 0.65rem; color: var(--text-faint); white-space: nowrap;
    display: flex; align-items: center; gap: 4px;
  }
  #controls input[type=range] { width: 64px; accent-color: var(--text-faint); cursor: pointer; opacity: 0.6; }
  #controls input[type=range]:hover { opacity: 1; }
  #controls .val { font-size: 0.65rem; color: var(--text-faint); min-width: 20px; }

  /* Footer info */
  #minfo { font-size: 0.65rem; color: var(--text-faint); padding: 0 16px 10px; flex-shrink: 0; }
  #minfo a { color: var(--text-faint); text-decoration: none; }
  #minfo a:hover { color: var(--text-secondary); }

  @media (max-width: 480px) {
    .msg { max-width: 90%; font-size: 0.92rem; }
    #header { padding: 10px 14px; }
    #messages { padding: 16px 12px; }
    #input-area { padding: 10px 12px 6px; }
  }
</style>
</head>
<body>
<div id="header">
  <div><div class="title">autoresearch</div><div class="sub">SFT chat · small model, be patient</div></div>
  <a href="/blog">training log →</a>
</div>
<div id="messages">
  <div id="empty">ask something — the model is small, answers will be rough</div>
</div>
<div id="input-area">
  <div id="input-row">
    <textarea id="input" placeholder="Ask something…" rows="1"></textarea>
    <button id="send" onclick="sendMsg()">↑</button>
  </div>
  <button id="controls-toggle" onclick="toggleControls()"><span class="arrow">▶</span> settings</button>
  <div id="controls">
    <label>temp <input type="range" id="temp" min="0" max="2" step="0.05" value="0.7"><span class="val" id="temp-val">0.7</span></label>
    <label>top-k <input type="range" id="topk" min="0" max="200" step="5" value="50"><span class="val" id="topk-val">50</span></label>
    <label>max <input type="range" id="maxtok" min="16" max="512" step="16" value="256"><span class="val" id="maxtok-val">256</span></label>
  </div>
</div>
<script>
let history = [], es = null;
const inputEl = document.getElementById('input');
const msgEl = document.getElementById('messages');
document.querySelectorAll('input[type=range]').forEach(el => {
  el.addEventListener('input', () => document.getElementById(el.id+'-val').textContent = el.value);
});
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
});
function toggleControls() {
  const btn = document.getElementById('controls-toggle');
  const ctrl = document.getElementById('controls');
  btn.classList.toggle('open');
  ctrl.classList.toggle('open');
}
function addMsg(role, text, id) {
  const empty = document.getElementById('empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (id) div.id = id;
  div.textContent = text;
  msgEl.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
  return div;
}
function sendMsg() {
  const text = inputEl.value.trim();
  if (!text || es) return;
  inputEl.value = ''; inputEl.style.height = 'auto';
  history.push({role: 'user', content: text});
  addMsg('user', text);
  const thinkingDiv = addMsg('assistant', '…', 'thinking');
  thinkingDiv.classList.add('thinking');
  document.getElementById('send').disabled = true;
  let accumulated = '';
  const p = new URLSearchParams({
    messages: JSON.stringify(history),
    temperature: document.getElementById('temp').value,
    top_k: document.getElementById('topk').value,
    max_tokens: document.getElementById('maxtok').value
  });
  es = new EventSource('/chat?' + p);
  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.done) {
      es.close(); es = null;
      document.getElementById('send').disabled = false;
      thinkingDiv.classList.remove('thinking');
      history.push({role: 'assistant', content: accumulated});
      return;
    }
    accumulated += d.token;
    thinkingDiv.textContent = accumulated;
    thinkingDiv.scrollIntoView({behavior: 'smooth', block: 'end'});
  };
  es.onerror = () => {
    es.close(); es = null;
    document.getElementById('send').disabled = false;
    thinkingDiv.textContent = '[error]';
  };
}
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
inputEl.focus();
</script>
</body>
</html>"""
CHAT_HTML = CHAT_HTML.replace("</body>", _footer_html + "\n</body>")


def make_stream(prompt_tokens, max_tokens, temperature, top_k, stop_token=None, repetition_penalty=1.3):
    accumulated = []
    def stream():
        for token_id in sample(prompt_tokens, max_tokens, temperature, top_k, stop_token, repetition_penalty):
            accumulated.append(token_id)
            text = enc.decode(accumulated)
            if not text.endswith("\ufffd"):
                prev = enc.decode(accumulated[:-1]) if len(accumulated) > 1 else ""
                new_text = text[len(prev):]
                if new_text:
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
        yield 'data: {"done": true}\n\n'
    return stream


@app.get("/", response_class=HTMLResponse)
async def root():
    return CHAT_HTML if args.sft else BASE_HTML


@app.get("/blog", response_class=HTMLResponse)
async def blog():
    import re
    blog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog.md")
    if not os.path.exists(blog_path):
        return HTMLResponse("<p>No blog yet.</p>", status_code=404)
    with open(blog_path) as _f:
        md_text = _f.read()

    def md_inline(line):
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        return line

    def render_block(lines):
        """Render a list of lines as HTML, handling tables and blockquotes."""
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("### "):
                out.append(f"<h3>{md_inline(line[4:])}</h3>")
            elif line.startswith("## "):
                out.append(f"<h2>{md_inline(line[3:])}</h2>")
            elif line.startswith("# "):
                out.append(f"<h1>{md_inline(line[2:])}</h1>")
            elif line.startswith("---"):
                pass  # skip HRs inside entries
            elif line.startswith("> "):
                out.append(f'<div class="response">{md_inline(line[2:])}</div>')
            elif line.startswith("|"):
                # table
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    rows.append(lines[i])
                    i += 1
                thtml = '<table>'
                for ri, row in enumerate(rows):
                    if re.match(r"^\|[-| ]+\|$", row):
                        continue  # separator row
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    tag = "th" if ri == 0 else "td"
                    thtml += "<tr>" + "".join(f"<{tag}>{md_inline(c)}</{tag}>" for c in cells) + "</tr>"
                thtml += "</table>"
                out.append(thtml)
                continue
            elif line.strip() == "":
                pass
            else:
                out.append(f"<p>{md_inline(line)}</p>")
            i += 1
        return "\n".join(out)

    # Split into: intro (before first ##) and entries (each ## block)
    parts = re.split(r"^## ", md_text, flags=re.MULTILINE)
    intro_md = parts[0]
    entry_mds = parts[1:]  # each starts with the header text

    # Render intro (skip the h1, hr lines cleanly)
    intro_lines = [l for l in intro_md.split("\n") if not l.startswith("---") and l.strip() != ""]
    intro_html = render_block(intro_lines)

    # Render entries — newest first
    entry_htmls = []
    for idx, entry_md in enumerate(reversed(entry_mds)):
        entry_lines = entry_md.split("\n")
        title = md_inline(entry_lines[0].strip())
        body_lines = entry_lines[1:]

        # Split body into main content and benchmark responses section
        bench_start = next((i for i, l in enumerate(body_lines) if "Benchmark responses" in l), None)
        if bench_start is not None:
            main_lines = body_lines[:bench_start]
            bench_lines = body_lines[bench_start:]
            bench_html = render_block(bench_lines)
            bench_section = f'<details class="bench"><summary>benchmark responses ▾</summary><div class="bench-body">{bench_html}</div></details>'
        else:
            main_lines = body_lines
            bench_section = ""

        main_html = render_block(main_lines)
        is_newest = (idx == 0)
        open_attr = " open" if is_newest else ""
        entry_htmls.append(
            f'<details class="entry"{open_attr}>'
            f'<summary class="entry-title">## {title}</summary>'
            f'<div class="entry-body">{main_html}{bench_section}</div>'
            f'</details>'
        )

    entries_html = "\n".join(entry_htmls)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch · training log</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
          background: #09090b; color: #d4d4d8; padding: 32px 20px 64px;
          max-width: 760px; margin: 0 auto; line-height: 1.75; font-size: 1rem; }}
  a.back {{ display: inline-block; margin-bottom: 24px; font-size: 0.82rem; color: #5b8ff4;
            text-decoration: none; }}
  a.back:hover {{ color: #88aaff; }}
  .intro {{ margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #1f1f23; }}
  .intro h1 {{ font-size: 1.4rem; color: #f0f0f2; margin-bottom: 14px; font-weight: 700; letter-spacing: -0.02em; }}
  .intro p {{ color: #b4b4be; font-size: 0.97rem; margin: 8px 0; line-height: 1.75; }}
  .intro strong {{ color: #d4d4d8; }}

  /* Entry accordion */
  details.entry {{ border: 1px solid #1f1f23; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }}
  details.entry[open] {{ border-color: #2a2a35; }}
  summary.entry-title {{
    list-style: none; cursor: pointer; padding: 14px 18px;
    font-size: 0.92rem; font-weight: 600; color: #a1a1aa;
    background: #111113; user-select: none;
    display: flex; align-items: center; gap: 8px;
  }}
  summary.entry-title::-webkit-details-marker {{ display: none; }}
  details.entry[open] summary.entry-title {{ color: #f0f0f2; border-bottom: 1px solid #1f1f23; }}
  summary.entry-title::before {{ content: "▶"; font-size: 0.6rem; color: #3f3f46; transition: transform 0.2s; }}
  details.entry[open] summary.entry-title::before {{ transform: rotate(90deg); color: #5b8ff4; }}

  .entry-body {{ padding: 18px 20px; }}
  .entry-body h3 {{ font-size: 0.72rem; font-weight: 700; color: #52525b; text-transform: uppercase;
                    letter-spacing: 0.07em; margin: 18px 0 6px; }}
  .entry-body p {{ color: #c4c4ce; font-size: 0.95rem; margin: 6px 0; line-height: 1.7; }}
  .entry-body strong {{ color: #e8e8f0; }}
  .entry-body em {{ color: #a1a1aa; }}
  .entry-body code {{ font-family: monospace; font-size: 0.82rem; background: #1a1a1f; padding: 1px 5px; border-radius: 3px; color: #a0c8ff; }}

  /* Table (quality assessment) */
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.82rem; }}
  th {{ text-align: left; color: #71717a; font-weight: 600; padding: 6px 10px;
        border-bottom: 1px solid #2a2a35; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1a1a1f; color: #a1a1aa; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}

  /* Benchmark responses nested accordion */
  details.bench {{ margin-top: 14px; border: 1px solid #1a1a1f; border-radius: 6px; overflow: hidden; }}
  details.bench summary {{ list-style: none; cursor: pointer; padding: 9px 14px;
    font-size: 0.75rem; color: #3f3f46; background: #0d0d0f; user-select: none; }}
  details.bench summary::-webkit-details-marker {{ display: none; }}
  details.bench[open] summary {{ color: #71717a; border-bottom: 1px solid #1a1a1f; }}
  .bench-body {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; }}
  .response {{ font-size: 0.82rem; color: #71717a; background: #0d0d0f; border-left: 2px solid #2a2a35;
               padding: 8px 12px; border-radius: 0 4px 4px 0; white-space: pre-wrap; word-break: break-word;
               font-style: italic; line-height: 1.6; }}
  .bench-body p {{ font-size: 0.8rem; color: #52525b; margin: 0; }}
  .bench-body em {{ color: #3f3f46; }}
</style>
</head>
<body>
<a class="back" href="/">← back to chat</a>
<div class="intro">{intro_html}</div>
{entries_html}
</body>
</html>"""


@app.get("/generate")
async def generate_endpoint(prompt: str, temperature: float = args.temperature,
                             top_k: int = args.top_k, max_tokens: int = args.max_tokens):
    temperature = max(0.0, min(2.0, temperature))
    top_k = max(0, min(200, top_k))
    max_tokens = max(1, min(1024, max_tokens))
    prompt_tokens = [bos_id] + enc.encode_ordinary(prompt)
    return StreamingResponse(make_stream(prompt_tokens, max_tokens, temperature, top_k)(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/chat")
async def chat_endpoint(messages: str, temperature: float = args.temperature,
                        top_k: int = args.top_k, max_tokens: int = args.max_tokens):
    temperature = max(0.0, min(2.0, temperature))
    top_k = max(0, min(200, top_k))
    max_tokens = max(1, min(1024, max_tokens))
    # Build prompt tokens from conversation history
    conversation = json.loads(messages)
    tokens = [bos_id]
    for msg in conversation:
        if msg["role"] == "user":
            tokens += [user_id] + enc.encode_ordinary(msg["content"]) + [end_id]
        elif msg["role"] == "assistant":
            tokens += [asst_id] + enc.encode_ordinary(msg["content"]) + [end_id]
    tokens.append(asst_id)  # prime the assistant response
    return StreamingResponse(make_stream(tokens, max_tokens, temperature, top_k, stop_token=end_id)(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    mode = "SFT chat" if args.sft else "base completion"
    print(f"\nStarting server ({mode}) at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
