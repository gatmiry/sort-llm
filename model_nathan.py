import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc_2 = nn.Linear(config.n_embd * 3, config.n_embd)

    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.stored_attn = None

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        attn = q @ k.transpose(-1, -2) / (k.size(-1)) ** 0.5
        attn = F.softmax(attn, dim=-1)
        self.stored_attn = attn.squeeze(0).squeeze(0)
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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        max_seq_len = getattr(config, 'max_seq_len', config.block_size)
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(max_seq_len, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
        )
        if config.use_final_LN:
            modules['ln_f'] = nn.LayerNorm(config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        if self.config.use_final_LN:
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class GPTConfig:
    block_size: int = 32
    vocab_size: int = 129
    n_layers: int = 2
    n_heads: int = 1
    n_embd: int = 64
    without_pos: bool = True
    use_mlp: bool = True
    use_final_LN: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
