"""
Shared SortGPT model definition and data utilities.

All experiment scripts import from here to avoid duplication.
"""

import math
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Device setup ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

try:
    BF16_OK = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
except Exception:
    BF16_OK = False
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float16


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_generator(device, seed):
    try:
        g = torch.Generator(device=device.type)
    except Exception:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def autocast_ctx(device, enabled=True):
    if (not enabled) or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=AMP_DTYPE)
    except Exception:
        return torch.cuda.amp.autocast(dtype=AMP_DTYPE)


def make_grad_scaler(enabled):
    if not enabled:
        class _NoScaler:
            def is_enabled(self): return False
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        return _NoScaler()
    try:
        return torch.amp.GradScaler()
    except Exception:
        return torch.cuda.amp.GradScaler()


def float_token(value):
    """Encode a float for use in filenames: 0.02 -> '0p02', -0.1 -> 'm0p1'."""
    return str(value).replace("-", "m").replace(".", "p")


# ── Data generation ──────────────────────────────────────────────────────────

def _sample_numbers(batch_size, vocab_n, length, device, allow_duplicates, *, generator=None):
    if allow_duplicates:
        return torch.randint(0, vocab_n, (batch_size, length), device=device,
                             generator=generator, dtype=torch.long)
    scores = torch.rand(batch_size, vocab_n, device=device, generator=generator)
    return scores.topk(length, dim=1).indices.to(torch.long)


def get_batch(batch_size, length, device, *, vocab_n, allow_duplicates=False, generator=None):
    """
    Generate a batch for the sorting task.

    Returns tensor of shape (batch_size, 2*length+1):
        [unsorted_tokens | SEP | sorted_tokens]

    SEP token = vocab_n (one above the max token value).
    """
    x = _sample_numbers(batch_size, vocab_n, length, device, allow_duplicates, generator=generator)
    vals = x.sort(dim=1).values
    sep = torch.full((batch_size, 1), vocab_n, device=device, dtype=torch.long)
    return torch.cat([x, sep, vals], dim=1)


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc_1 = nn.Linear(n_embd, 3 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc_2 = nn.Linear(3 * n_embd, n_embd)

    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = int(n_embd // n_heads)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers, use_mlp=True):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_heads, n_layers)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.use_mlp = bool(use_mlp)
        if self.use_mlp:
            self.mlp = MLP(n_embd)
            self.ln_2 = nn.LayerNorm(n_embd)
        else:
            self.mlp = None
            self.ln_2 = None

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    without_pos: bool
    use_mlp: bool
    use_final_LN: bool
    max_seq_len: int


class GPT(nn.Module):
    _init_std = 0.02  # Set before __init__ to control initialization scale

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = int(config.n_layers)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.max_seq_len, config.n_embd),
            h=nn.ModuleList([
                Block(config.n_embd, config.n_heads, config.n_layers, use_mlp=config.use_mlp)
                for _ in range(config.n_layers)
            ]),
            ln_f=(nn.LayerNorm(config.n_embd) if config.use_final_LN else nn.Identity()),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        self.register_buffer("pos_idx", torch.arange(config.max_seq_len), persistent=False)
        if config.without_pos:
            with torch.no_grad():
                self.transformer.wpe.weight.zero_()
            self.transformer.wpe.weight.requires_grad_(False)

    def _init_weights(self, module):
        std = self.__class__._init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, idx, *, block_size, return_full_logits=False):
        B, T = idx.size()
        expected_T = 2 * int(block_size) + 1
        assert T == expected_T, f"Expected T={expected_T}, got T={T}"
        assert T <= self.config.max_seq_len, f"T={T} exceeds max_seq_len={self.config.max_seq_len}"
        pos = self.transformer.wpe(self.pos_idx[:T])
        x = self.transformer.wte(idx) if self.config.without_pos else (self.transformer.wte(idx) + pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits_half = self.lm_head(x[:, block_size:T - 1, :])
        targets = idx[:, block_size + 1:]
        loss = F.cross_entropy(logits_half.reshape(-1, logits_half.size(-1)), targets.reshape(-1))
        if return_full_logits:
            return self.lm_head(x), loss
        return logits_half, loss


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path, *, extended_max_seq_len=None):
    """
    Load a model from a checkpoint file.

    Args:
        ckpt_path: Path to .pt checkpoint file.
        extended_max_seq_len: If set, extend the positional embedding table
            to support longer sequences at eval time. Only works when
            without_pos=True (pos embeddings are zeroed).

    Returns:
        model: GPT model on DEVICE in eval mode.
    """
    artifact = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = artifact["model_config"]
    model_cfg = GPTConfig(**cfg_dict)
    model = GPT(model_cfg)
    model.load_state_dict(artifact["model_state_dict"])

    if extended_max_seq_len and extended_max_seq_len > cfg_dict["max_seq_len"]:
        model.config = GPTConfig(**dict(cfg_dict, max_seq_len=extended_max_seq_len))
        new_wpe = nn.Embedding(extended_max_seq_len, model_cfg.n_embd)
        with torch.no_grad():
            new_wpe.weight.zero_()
        new_wpe.weight.requires_grad_(False)
        model.transformer.wpe = new_wpe
        model.register_buffer("pos_idx", torch.arange(extended_max_seq_len), persistent=False)

    return model.to(DEVICE).eval()


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(itr, max_iters, learning_rate, warmup_iters, min_lr):
    """Cosine decay with linear warmup."""
    if itr < warmup_iters:
        return learning_rate * (itr + 1) / (warmup_iters + 1)
    if itr >= max_iters:
        return min_lr
    ratio = (itr - warmup_iters) / max(max_iters - warmup_iters, 1)
    ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + ratio * (learning_rate - min_lr)


def create_optimizer(model, *, weight_decay, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if DEVICE.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                     weight_decay=float(weight_decay), fused=True)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=float(weight_decay))
