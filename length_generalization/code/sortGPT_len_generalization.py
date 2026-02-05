#!/usr/bin/env python3
import os, math, time, argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from fractions import Fraction
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

import wandb


# =========================
# Speed knobs / kernels
# =========================
def enable_tf32():
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True

        if (
            hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "conv")
            and hasattr(torch.backends.cudnn.conv, "fp32_precision")
        ):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_sdpa_context():
    """
    Returns a real context-manager object (not a generator).
    Uses torch.nn.attention.sdpa_kernel if available; otherwise no-op.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except Exception:
        return nullcontext()


def get_autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    """
    Returns a real context-manager object for autocast.
    Never passes device_type=... (compat across torch versions).
    """
    if device.type != "cuda" or dtype is None:
        return nullcontext()

    try:
        return torch.amp.autocast("cuda", dtype=dtype)  # positional device arg
    except Exception:
        return torch.cuda.amp.autocast(dtype=dtype)     # no device_type kwarg


def make_grad_scaler(enabled: bool):
    """
    Returns a scaler-like object with .is_enabled(), .scale(), .step(), .update().
    """
    if not enabled:
        class _NoScaler:
            def is_enabled(self): return False
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
        return _NoScaler()

    try:
        return torch.amp.GradScaler()
    except Exception:
        return torch.cuda.amp.GradScaler()


# =========================
# Model components
# =========================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc_2 = nn.Linear(3 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_mlp = bool(getattr(config, "use_mlp", True))
        self.use_layernorm = bool(getattr(config, "use_layernorm", True))

        self.attn = CausalSelfAttention(config)
        if self.use_layernorm:
            self.ln_1 = nn.LayerNorm(config.n_embd)
        else:
            self.ln_1 = nn.Identity()

        if self.use_mlp:
            self.mlp = MLP(config)
            if self.use_layernorm:
                self.ln_2 = nn.LayerNorm(config.n_embd)
            else:
                self.ln_2 = nn.Identity()
        else:
            self.mlp = None
            self.ln_2 = None

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 1,
        n_embd: int = 128,
        without_pos: bool = False,
        use_mlp: bool = True,
        use_layernorm: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_embd = int(n_embd)
        self.without_pos = bool(without_pos)
        self.use_mlp = bool(use_mlp)
        self.use_layernorm = bool(use_layernorm)
        self.max_seq_len = int(max_seq_len if max_seq_len is not None else (2 * self.block_size + 1))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers

        use_layernorm = bool(getattr(config, "use_layernorm", True))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.max_seq_len, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.n_embd) if use_layernorm else nn.Identity(),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying
        self.apply(self._init_weights)

        self.register_buffer("pos_idx", torch.arange(config.max_seq_len), persistent=False)

        if self.config.without_pos:
            with torch.no_grad():
                self.transformer.wpe.weight.zero_()
            self.transformer.wpe.weight.requires_grad_(False)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, return_full_logits: bool = False, block_size: Optional[int] = None):
        B, T = idx.size()
        if block_size is None:
            block_size = self.config.block_size
        block_size = int(block_size)

        expected_T = 2 * block_size + 1
        assert T == expected_T, f"Expected T={expected_T} for block_size={block_size}, got T={T}"
        assert T <= self.config.max_seq_len, f"T={T} exceeds max_seq_len={self.config.max_seq_len}"

        pos = self.transformer.wpe(self.pos_idx[:T])
        x = self.transformer.wte(idx) if self.config.without_pos else (self.transformer.wte(idx) + pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        targets = idx[:, block_size + 1 :]  # (B, K)

        if return_full_logits:
            logits = self.lm_head(x)                        # (B, T, V)
            logits_for_loss = logits[:, block_size:T-1, :]  # (B, K, V)
        else:
            x_for_loss = x[:, block_size:T-1, :]            # (B, K, C)
            logits_for_loss = self.lm_head(x_for_loss)      # (B, K, V)
            logits = logits_for_loss

        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss


# =========================
# Data batching (NoDup enforced)
# =========================
SEP_TOKEN = "SEP"

def _sample_numbers(batch_size: int, vocab_n: int, block_size: int, device: torch.device, allow_duplicates: bool):
    if allow_duplicates:
        return torch.randint(0, vocab_n, (batch_size, block_size), device=device, dtype=torch.long)

    if block_size > vocab_n:
        raise ValueError(f"allow_duplicates=False requires block_size <= vocab_n (got block_size={block_size}, vocab_n={vocab_n})")

    scores = torch.rand(batch_size, vocab_n, device=device)
    return scores.topk(block_size, dim=1).indices.to(torch.long)

def get_batch(batch_size: int, device: torch.device, vocab_n: int, block_size: int, allow_duplicates: bool):
    x = _sample_numbers(batch_size, vocab_n, block_size, device, allow_duplicates)
    vals = x.sort(dim=1).values
    sep_id = vocab_n
    sep = torch.full((batch_size, 1), sep_id, device=device, dtype=torch.long)
    return torch.cat([x, sep, vals], dim=1)


# =========================
# LR helpers
# =========================
def create_optimizer(model, weight_decay: float, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() > 1]
    nondecay_params = [p for p in params if p.dim() <= 1]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0},
    ]
    try:
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    except TypeError:
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

def get_lr(itr: int, cfg) -> float:
    if itr < cfg.warmup_iters:
        return cfg.learning_rate * (itr + 1) / (cfg.warmup_iters + 1)
    if itr > cfg.max_iters:
        return cfg.min_lr
    ratio = (itr - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + ratio * (cfg.learning_rate - cfg.min_lr)


# =========================
# Metrics
# =========================
@torch.no_grad()
def eval_length_metrics(
    model: nn.Module,
    device: torch.device,
    cfg,
    block_size: int,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    num_samples: int,
):
    """
    Returns:
      loss_mean, exact_match_acc, token_acc, pos_abs_err_sum, pos_abs_err_mean
    where pos_abs_err_sum = sum_{samples, positions} |pred - target|
    """
    model.eval()

    remaining = int(num_samples)
    loss_sum = 0.0
    n_batches = 0

    exact_match_correct = 0
    token_correct = 0
    token_total = 0

    abs_err_sum = 0.0

    while remaining > 0:
        bs = min(int(cfg.eval_batch_size), remaining)
        remaining -= bs

        batch = get_batch(
            batch_size=bs,
            device=device,
            vocab_n=cfg.vocab_n,
            block_size=int(block_size),
            allow_duplicates=cfg.allow_duplicates,
        )

        with (get_autocast_context(device, amp_dtype) if use_amp else nullcontext()):
            logits, loss = model(batch, return_full_logits=False, block_size=int(block_size))

        loss_sum += float(loss.item())
        n_batches += 1

        targets = batch[:, int(block_size) + 1 :]
        preds = logits.argmax(dim=-1)

        exact_match_correct += int((preds == targets).all(dim=1).sum().item())
        token_correct += int((preds == targets).sum().item())
        token_total += int(targets.numel())

        abs_err_sum += float((preds.to(torch.long) - targets.to(torch.long)).abs().sum().item())

    loss_mean = loss_sum / max(n_batches, 1)
    exact_match_acc = exact_match_correct / max(num_samples, 1)
    token_acc = token_correct / max(token_total, 1)
    pos_abs_err_mean = abs_err_sum / max(token_total, 1)

    model.train()
    return loss_mean, exact_match_acc, token_acc, abs_err_sum, pos_abs_err_mean


# =========================
# Config
# =========================
@dataclass
class TrainConfig:
    # Fixed by plan (but still CLI configurable if you really want)
    vocab_n: int = 128
    n_embd: int = 128
    n_heads: int = 1

    # Train lengths / test lengths
    train_min_k: int = 2
    train_max_k: int = 16
    test_min_k: int = 2
    test_max_k: int = 32

    # Data constraint
    allow_duplicates: bool = False  # NoDup only

    # Model
    n_layers: int = 2
    without_pos: bool = False
    use_mlp: bool = True
    use_layernorm: bool = True

    # Length sampling
    length_mode: str = "mix"  # mix | curriculum
    curriculum_warmup_iters: int = 0
    curriculum_span_iters: int = 20000  # how long to go from min_k -> max_k

    # Training recipe
    warmup_iters: int = 200
    max_iters: int = 25000
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0

    micro_batch_size: int = 4096
    effective_batch_size: int = 4096

    log_interval: int = 250
    eval_interval: int = 250
    ckpt_interval: int = 20000
    save_dir: str = "./saved_models"

    # Eval protocol
    eval_samples_per_length: int = 100
    eval_batch_size: int = 100  # batch size used during evaluation loops

    seed: int = 1337
    use_compile: bool = False

    wandb_project: str = "sortgpt"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_mode: Optional[str] = None


# =========================
# Naming / saving
# =========================
def _block_range_tag(a: int, b: int) -> str:
    return f"K{int(a)}to{int(b)}"

def make_wandb_run_name(cfg: TrainConfig) -> str:
    ga = int(cfg.effective_batch_size) // int(cfg.micro_batch_size)
    return (
        f"bs{int(cfg.micro_batch_size)}"
        f"_eb{int(cfg.effective_batch_size)}"
        f"_ga{ga}"
        f"_N{int(cfg.vocab_n)}"
        f"_d{int(cfg.n_embd)}"
        f"_H{int(cfg.n_heads)}"
        f"_L{int(cfg.n_layers)}"
        f"_npos{int(cfg.without_pos)}"
        f"_mlp{int(cfg.use_mlp)}"
        f"_ln{int(cfg.use_layernorm)}"
        f"_len{cfg.length_mode}"
        f"_train{_block_range_tag(cfg.train_min_k, cfg.train_max_k)}"
        f"_test{_block_range_tag(cfg.test_min_k, cfg.test_max_k)}"
        f"_nodup{int(not cfg.allow_duplicates)}"
    )

def make_save_filename(prefix: str, cfg: TrainConfig, iters_done: int) -> str:
    return (
        f"{prefix}"
        f"_N{int(cfg.vocab_n)}"
        f"_d{int(cfg.n_embd)}"
        f"_H{int(cfg.n_heads)}"
        f"_L{int(cfg.n_layers)}"
        f"_npos{int(cfg.without_pos)}"
        f"_mlp{int(cfg.use_mlp)}"
        f"_ln{int(cfg.use_layernorm)}"
        f"_len{cfg.length_mode}"
        f"_train{_block_range_tag(cfg.train_min_k, cfg.train_max_k)}"
        f"_test{_block_range_tag(cfg.test_min_k, cfg.test_max_k)}"
        f"_nodup{int(not cfg.allow_duplicates)}"
        f"_iters{int(iters_done)}.pt"
    )


# =========================
# Length schedule
# =========================
def pick_train_k(itr: int, cfg: TrainConfig) -> int:
    """
    mix: uniform K in [train_min_k, train_max_k]
    curriculum: max allowed K increases from min->max over curriculum_span_iters (after curriculum_warmup_iters)
    """
    k0 = int(cfg.train_min_k)
    k1 = int(cfg.train_max_k)
    assert k0 <= k1

    if cfg.length_mode == "mix":
        return int(torch.randint(low=k0, high=k1 + 1, size=(1,)).item())

    if cfg.length_mode == "curriculum":
        t = max(0, int(itr) - int(cfg.curriculum_warmup_iters))
        span = max(1, int(cfg.curriculum_span_iters))
        frac = min(1.0, t / span)
        max_k = k0 + int(round(frac * (k1 - k0)))
        max_k = max(k0, min(k1, max_k))
        return int(torch.randint(low=k0, high=max_k + 1, size=(1,)).item())

    raise ValueError(f"Unknown length_mode: {cfg.length_mode} (expected mix|curriculum)")


# =========================
# Training
# =========================
def train_sorting_gpt(cfg: TrainConfig):
    # Enforce FixedData / NoDup only
    if cfg.allow_duplicates:
        raise ValueError("This experiment plan is NoDup only. Set allow_duplicates=False.")

    if cfg.n_heads != 1:
        raise ValueError("Plan fixes n_heads=1. Please set --n-heads 1.")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    total_vocab_size = cfg.vocab_n + 1
    sep_id = cfg.vocab_n

    grad_accum_steps = cfg.effective_batch_size // cfg.micro_batch_size
    assert cfg.effective_batch_size % cfg.micro_batch_size == 0

    use_amp = (device.type == "cuda")
    if use_amp:
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        amp_dtype = None

    scaler = make_grad_scaler(enabled=(use_amp and amp_dtype == torch.float16))

    # model must support up to test_max_k
    max_k_for_model = int(cfg.test_max_k)
    max_seq_len = 2 * max_k_for_model + 1

    # block_size here is just a default; we always pass block_size explicitly in forward
    model_cfg = GPTConfig(
        block_size=int(cfg.train_max_k),
        vocab_size=total_vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_embd=cfg.n_embd,
        without_pos=cfg.without_pos,
        use_mlp=cfg.use_mlp,
        use_layernorm=cfg.use_layernorm,
        max_seq_len=max_seq_len,
    )
    model = GPT(model_cfg).to(device)

    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, continuing uncompiled: {e}")

    optimizer = create_optimizer(model, weight_decay=cfg.weight_decay, lr=cfg.learning_rate)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if wandb.run is not None:
        wandb.finish()

    wandb_cfg = asdict(cfg)
    wandb_cfg.update(
        dict(
            total_vocab_size=total_vocab_size,
            sep_id=sep_id,
            max_seq_len=max_seq_len,
            grad_accum_steps=grad_accum_steps,
            amp_dtype=str(amp_dtype) if amp_dtype is not None else "none",
            device=str(device),
        )
    )

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=make_wandb_run_name(cfg),
        config=wandb_cfg,
        mode=cfg.wandb_mode,
    )

    run.define_metric("iter")
    run.define_metric("train/*", step_metric="iter")
    run.define_metric("test/*", step_metric="iter")
    run.define_metric("gen/*", step_metric="iter")
    run.define_metric("lr", step_metric="iter")

    iters_done = cfg.max_iters
    last_log_t = time.time()

    def do_full_eval(step: int):
        # In-dist: train_min_k..train_max_k
        for k in range(int(cfg.train_min_k), int(cfg.train_max_k) + 1):
            loss, em_acc, tok_acc, abs_sum, abs_mean = eval_length_metrics(
                model=model,
                device=device,
                cfg=cfg,
                block_size=int(k),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                num_samples=int(cfg.eval_samples_per_length),
            )
            run.log(
                {
                    "iter": step,
                    f"test/K{k}/loss": loss,
                    f"test/K{k}/exact_match_acc": em_acc,
                    f"test/K{k}/token_acc": tok_acc,
                    f"test/K{k}/pos_abs_err_sum": abs_sum,
                    f"test/K{k}/pos_abs_err_mean": abs_mean,
                },
                step=step,
            )

        # Gen: max(train_max_k + 1, test_min_k)..test_max_k
        gen_start = max(int(cfg.train_max_k) + 1, int(cfg.test_min_k))
        for k in range(gen_start, int(cfg.test_max_k) + 1):
            loss, em_acc, tok_acc, abs_sum, abs_mean = eval_length_metrics(
                model=model,
                device=device,
                cfg=cfg,
                block_size=int(k),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                num_samples=int(cfg.eval_samples_per_length),
            )
            run.log(
                {
                    "iter": step,
                    f"gen/K{k}/loss": loss,
                    f"gen/K{k}/exact_match_acc": em_acc,
                    f"gen/K{k}/token_acc": tok_acc,
                    f"gen/K{k}/pos_abs_err_sum": abs_sum,
                    f"gen/K{k}/pos_abs_err_mean": abs_mean,
                },
                step=step,
            )

    with get_sdpa_context():
        for itr in trange(cfg.max_iters, desc="training"):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)

            # train logging (on the sampled K values)
            do_log = (itr % cfg.log_interval == 0)
            token_correct = torch.zeros((), device=device) if do_log else None
            sample_correct = torch.zeros((), device=device) if do_log else None
            token_total = 0
            sample_total = 0
            k_hist = [] if do_log else None

            for _ in range(grad_accum_steps):
                k_train = pick_train_k(itr, cfg)
                if do_log:
                    k_hist.append(int(k_train))

                batch = get_batch(
                    batch_size=cfg.micro_batch_size,
                    device=device,
                    vocab_n=cfg.vocab_n,
                    block_size=int(k_train),
                    allow_duplicates=cfg.allow_duplicates,
                )

                if use_amp:
                    with get_autocast_context(device, amp_dtype):
                        logits, loss = model(batch, return_full_logits=False, block_size=int(k_train))
                else:
                    logits, loss = model(batch, return_full_logits=False, block_size=int(k_train))

                if do_log:
                    with torch.no_grad():
                        targets = batch[:, int(k_train) + 1 :]
                        preds = logits.detach().argmax(dim=-1)
                        token_correct += (preds == targets).sum()
                        sample_correct += (preds == targets).all(dim=1).sum()
                        token_total += targets.numel()
                        sample_total += targets.size(0)

                loss_to_back = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()

                loss_accum += loss.detach()

            lr = get_lr(itr, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            steps_done = itr + 1

            if do_log:
                train_loss = float((loss_accum / grad_accum_steps).item())
                train_token_acc = float((token_correct / max(token_total, 1)).item())
                train_exact_match_acc = float((sample_correct / max(sample_total, 1)).item())
                k_mean = float(sum(k_hist) / max(len(k_hist), 1))
                k_min = int(min(k_hist)) if len(k_hist) > 0 else -1
                k_max = int(max(k_hist)) if len(k_hist) > 0 else -1

                now = time.time()
                dt = now - last_log_t
                last_log_t = now
                print(
                    f"itr: {itr} lr: {lr:.3e} "
                    f"train loss: {train_loss:.6f} "
                    f"train token_acc: {train_token_acc:.4f} "
                    f"train exact_match_acc: {train_exact_match_acc:.4f} "
                    f"(K~{k_mean:.2f}, min={k_min}, max={k_max}, dt={dt:.2f}s)"
                )

                run.log(
                    {
                        "iter": steps_done,
                        "lr": lr,
                        "train/loss": train_loss,
                        "train/token_acc": train_token_acc,
                        "train/exact_match_acc": train_exact_match_acc,
                        "train/K_mean": k_mean,
                        "train/K_min": k_min,
                        "train/K_max": k_max,
                    },
                    step=steps_done,
                )

            if cfg.eval_interval and (itr % cfg.eval_interval == 0):
                do_full_eval(steps_done)

            if cfg.ckpt_interval and (itr > 0) and (itr % cfg.ckpt_interval == 0):
                ckpt_name = make_save_filename("Checkpoint", cfg, steps_done)
                ckpt_path = os.path.join(cfg.save_dir, ckpt_name)
                torch.save(
                    {
                        "itr": itr,
                        "iters_done": steps_done,
                        "train_config": wandb_cfg,
                        "model_config": vars(model_cfg),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

    final_name = make_save_filename("Final", cfg, iters_done)
    final_path = os.path.join(cfg.save_dir, final_name)
    torch.save(
        {
            "iters_done": iters_done,
            "train_config": wandb_cfg,
            "model_config": vars(model_cfg),
            "model": model.state_dict(),
        },
        final_path,
    )
    print(f"saved final model: {final_path}")

    run.summary["iters_done"] = iters_done
    run.finish()
    return final_path


# =========================
# CLI / Grid helpers
# =========================
def parse_bool_list(xs: List[str]) -> List[bool]:
    out = []
    for x in xs:
        s = str(x).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            out.append(True)
        elif s in ("0", "false", "f", "no", "n", "off"):
            out.append(False)
        else:
            raise ValueError(f"Could not parse bool value: {x}")
    return out

def parse_str_list(xs: List[str]) -> List[str]:
    return [str(x).strip() for x in xs]

def build_grid(
    layer_counts: List[int],
    without_pos_flags: List[bool],
    use_mlp_flags: List[bool],
    use_layernorm_flags: List[bool],
    length_modes: List[str],
):
    layer_counts = sorted(list(layer_counts), reverse=True)

    norm_modes = []
    for m in length_modes:
        mm = str(m).strip().lower()
        if mm not in ("mix", "curriculum"):
            raise ValueError(f"Unknown length mode: {m} (expected mix|curriculum)")
        norm_modes.append(mm)

    combos = []
    for L in layer_counts:
        for npos in without_pos_flags:
            for mlp_on in use_mlp_flags:
                for ln_on in use_layernorm_flags:
                    for lm in norm_modes:
                        combos.append((int(L), bool(npos), bool(mlp_on), bool(ln_on), str(lm)))
    return combos

def get_task_id(cli_task_id: Optional[int]) -> int:
    if cli_task_id is not None:
        return int(cli_task_id)
    env = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if env is None:
        return 0
    return int(env)

def setup_run_dirs(root: str, group: str, task_id: int):
    os.makedirs(root, exist_ok=True)
    grid_root = os.path.join(root, group, f"task_{task_id:04d}")
    wandb_dir = os.path.join(grid_root, "wandb")
    save_dir  = os.path.join(grid_root, "saved_models")
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = wandb_dir
    os.environ["WANDB_CACHE_DIR"] = os.path.join(grid_root, "wandb_cache")
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(grid_root, "wandb_config")
    return grid_root, wandb_dir, save_dir


def main():
    enable_tf32()

    p = argparse.ArgumentParser("Grid train SortGPT w/ length generalization tests (one config per SLURM array task)")

    # WandB / output
    p.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "sortgpt"))
    p.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--group", type=str, default=None, help="If omitted, uses grid_<timestamp>")
    p.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", None), help="e.g. online|offline|disabled")
    p.add_argument("--root", type=str, default="./grid_outputs", help="Root directory for per-task outputs")
    p.add_argument("--task-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID (for local testing)")

    # Fixed-by-plan (but explicit here for safety)
    p.add_argument("--vocab-n", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=1)

    # Base train knobs
    p.add_argument("--max-iters", type=int, default=25000)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--micro-batch-size", type=int, default=4096)
    p.add_argument("--effective-batch-size", type=int, default=4096)
    p.add_argument("--log-interval", type=int, default=250)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--ckpt-interval", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--use-compile", action="store_true")

    # Train/test lengths
    p.add_argument("--train-min-k", type=int, default=2)
    p.add_argument("--train-max-k", type=int, default=16)
    p.add_argument("--test-min-k", type=int, default=2)
    p.add_argument("--test-max-k", type=int, default=32)

    # Length mode
    p.add_argument("--length-modes", type=str, nargs="+", required=True, help="mix curriculum")
    p.add_argument("--curriculum-warmup-iters", type=int, default=0)
    p.add_argument("--curriculum-span-iters", type=int, default=20000)

    # Eval protocol
    p.add_argument("--eval-samples-per-length", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=100)

    # Grid dimensions
    p.add_argument("--layer-counts", type=int, nargs="+", required=True)
    p.add_argument("--without-pos-flags", type=str, nargs="+", required=True, help="e.g. 0 1 or false true")
    p.add_argument("--use-mlp-flags", type=str, nargs="+", required=True, help="e.g. true false")
    p.add_argument("--use-layernorm-flags", type=str, nargs="+", default=["true"], help="e.g. true false")

    # NoDup: keep flag but enforce false
    p.add_argument("--allow-duplicates-flags", type=str, nargs="+", default=["false"])

    args = p.parse_args()

    without_pos_flags = parse_bool_list(args.without_pos_flags)
    use_mlp_flags = parse_bool_list(args.use_mlp_flags)
    use_layernorm_flags = parse_bool_list(args.use_layernorm_flags)
    allow_duplicates_flags = parse_bool_list(args.allow_duplicates_flags)
    length_modes = parse_str_list(args.length_modes)

    # Enforce NoDup-only plan
    if any(bool(x) for x in allow_duplicates_flags):
        raise ValueError("This experiment plan is NoDup only. Use --allow-duplicates-flags false")

    if int(args.n_heads) != 1:
        raise ValueError("Plan fixes n_heads=1. Please set --n-heads 1.")

    if int(args.vocab_n) != 128 or int(args.n_embd) != 128:
        print("⚠️  Plan expects vocab_n=128 and n_embd=128. Proceeding with provided values.")

    # group naming
    stamp = time.strftime("%Y%m%d_%H%M%S")
    group = args.group if args.group is not None else f"grid_{stamp}"

    task_id = get_task_id(args.task_id)

    combos = build_grid(
        layer_counts=args.layer_counts,
        without_pos_flags=without_pos_flags,
        use_mlp_flags=use_mlp_flags,
        use_layernorm_flags=use_layernorm_flags,
        length_modes=length_modes,
    )

    if len(combos) == 0:
        raise RuntimeError("Grid is empty. Check your --* lists.")

    if task_id < 0 or task_id >= len(combos):
        raise RuntimeError(
            f"Task id {task_id} is out of range for grid of size {len(combos)}. "
            f"Set --array=0-{len(combos)-1} in sbatch."
        )

    L, npos, mlp_on, ln_on, len_mode = combos[task_id]

    # output dirs (per task)
    grid_root, wandb_dir, save_dir = setup_run_dirs(args.root, group, task_id)

    print("======== GRID SELECTION ========")
    print(f"task_id      = {task_id}")
    print(f"grid_size    = {len(combos)}")
    print(f"grid_root    = {grid_root}")
    print(f"WANDB_DIR    = {wandb_dir}")
    print(f"SAVE_DIR     = {save_dir}")
    print("chosen config:")
    print(f"  n_layers        = {L}")
    print(f"  without_pos     = {npos}")
    print(f"  use_mlp         = {mlp_on}")
    print(f"  use_layernorm   = {ln_on}")
    print(f"  length_mode     = {len_mode}")
    print("================================")

    cfg = TrainConfig(
        vocab_n=int(args.vocab_n),
        n_embd=int(args.n_embd),
        n_heads=int(args.n_heads),

        n_layers=int(L),
        without_pos=bool(npos),
        use_mlp=bool(mlp_on),
        use_layernorm=bool(ln_on),

        allow_duplicates=False,

        train_min_k=int(args.train_min_k),
        train_max_k=int(args.train_max_k),
        test_min_k=int(args.test_min_k),
        test_max_k=int(args.test_max_k),

        length_mode=str(len_mode),
        curriculum_warmup_iters=int(args.curriculum_warmup_iters),
        curriculum_span_iters=int(args.curriculum_span_iters),

        max_iters=int(args.max_iters),
        warmup_iters=int(args.warmup_iters),
        learning_rate=float(args.learning_rate),
        min_lr=float(args.min_lr),
        weight_decay=float(args.weight_decay),

        micro_batch_size=int(args.micro_batch_size),
        effective_batch_size=int(args.effective_batch_size),

        log_interval=int(args.log_interval),
        eval_interval=int(args.eval_interval),
        ckpt_interval=int(args.ckpt_interval),
        save_dir=str(save_dir),

        eval_samples_per_length=int(args.eval_samples_per_length),
        eval_batch_size=int(args.eval_batch_size),

        seed=int(args.seed),
        use_compile=bool(args.use_compile),

        wandb_project=str(args.project),
        wandb_entity=args.entity,
        wandb_group=str(group),
        wandb_mode=args.wandb_mode,
    )

    train_sorting_gpt(cfg)


if __name__ == "__main__":
    main()