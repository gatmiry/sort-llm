"""
Model matching the 200k-checkpoints architecture exactly.
Block uses self.attn / self.mlp naming (matching 200k state dict).
max_seq_len configurable (200k model uses 193).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc_2 = nn.Linear(config.n_embd * 3, config.n_embd)
        self.NANO_SCALE_GPT = True

    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        seq_len = config.max_seq_len
        self.register_buffer('bias', torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
        self.c_proj.NANOGPT_SCALE_INIT = True
        self.config = config

    def forward(self, x, layer_n=-1):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, layer_n=-1):
        x = x + self.attn(self.ln_1(x), layer_n=layer_n)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
            wpe=nn.Embedding(config.max_seq_len, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, idx, targets=None, flag=False):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        layer_n = 0
        for block in self.transformer.h:
            layer_n += 1
            x = block(x, layer_n)
        if self.config.with_layer_norm:
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        tensor1 = logits[:, self.config.block_size:T - 1, :].contiguous().view(-1, logits.size(-1))
        tensor2 = idx[:, self.config.block_size + 1:].contiguous().view(-1)
        loss = F.cross_entropy(tensor1, tensor2)
        return logits, loss


class GPTConfig:
    block_size: int = 16
    vocab_size: int = 256
    n_layers: int = 2
    n_heads: int = 1
    n_embd: int = 64
    with_layer_norm: bool = True
    max_seq_len: int = 193

    def __init__(self, block_size=None, vocab_size=None, with_layer_norm=True, max_seq_len=193):
        if block_size is not None:
            self.block_size = block_size
        if vocab_size is not None:
            self.vocab_size = vocab_size
        self.with_layer_norm = with_layer_norm
        self.max_seq_len = max_seq_len
