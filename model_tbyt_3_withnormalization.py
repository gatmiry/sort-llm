import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Optional, Tuple, Dict, Any

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
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
        
        # Store attention weights for analysis
        self.attn_weights = None

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Manual attention computation to store weights
        # Grid training uses F.scaled_dot_product_attention which uses standard 1.0 / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Store attention weights (squeeze to 2D for single head, single batch access)
        # Shape: (B, n_heads, T, T) -> store as (T, T) for compatibility with statistics script
        self.attn_weights = attn[0, 0].detach()  # First batch, first head
        
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_mlp = getattr(config, "use_mlp", True)
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if self.use_mlp:
            self.mlp = MLP(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
        else:
            self.mlp = None
            self.ln_2 = None

    def forward(self, x, layer_n=-1):
        # For layer 1 (second layer, 0-indexed), skip the attention layer
        if layer_n == 1:
            x = x + self.attn(self.ln_1(x))  # Commented out for layer 1
            #pass
        else:
            x = x + self.attn(self.ln_1(x))
        if self.mlp is not None:
            if layer_n == 0:
                #pass
                x = x + self.mlp(self.ln_2(x))
            else:
                x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 1,
        n_embd: int = 64,
        without_pos: bool = False,
        use_mlp: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_embd = int(n_embd)
        self.without_pos = bool(without_pos)
        self.use_mlp = bool(use_mlp)
        self.max_seq_len = int(max_seq_len if max_seq_len is not None else (2 * self.block_size + 1))

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying
        
        self.register_buffer("pos_idx", torch.arange(config.max_seq_len), persistent=False)

        if self.config.without_pos:
            with torch.no_grad():
                self.transformer.wpe.weight.zero_()
            self.transformer.wpe.weight.requires_grad_(False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None, block_size: Optional[int] = None):
        B, T = idx.size()
        if block_size is None:
            block_size = self.config.block_size
        
        pos = self.transformer.wpe(self.pos_idx[:T])
        x = self.transformer.wte(idx) if self.config.without_pos else (self.transformer.wte(idx) + pos)

        for layer_n, block in enumerate(self.transformer.h):
            x = block(x, layer_n=layer_n)

        x = self.transformer.ln_f(x) # Final layer normalization

        # Always return full logits (like model_tbyt_3.py)
        logits = self.lm_head(x)  # (B, T, V)

        # Compute loss on the output portion only
        logits_for_loss = logits[:, block_size:T-1, :].contiguous().view(-1, logits.size(-1))
        targets_for_loss = idx[:, block_size + 1:].contiguous().view(-1)
        
        loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        
        return logits, loss

    def generate(self, idx, topk, sampling_length):
        self.eval()
        for _ in range(sampling_length):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            vals, indices = torch.topk(logits, topk, dim=-1)
            logits[logits < vals[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sampled_indices), dim=-1)
        self.train()
        return idx

    def proj_on_embedding(self, idx, index):
        B, T = idx.size()
        device = idx.device
        self.layer_probes = torch.zeros(B, self.config.n_layers + 1, self.config.vocab_size).to(device)
        with torch.no_grad():
            pos = self.transformer.wpe(self.pos_idx[:T])
            x = self.transformer.wte(idx) if self.config.without_pos else (self.transformer.wte(idx) + pos)
            
            # Initial projection (after embedding)
            self.layer_probes[:, 0, :] = self.lm_head(self.transformer.ln_f(x[:, index, :]))
            
            for depth, block in enumerate(self.transformer.h, start=1):
                x = block(x)
                # Projection after each block
                self.layer_probes[:, depth, :] = self.lm_head(self.transformer.ln_f(x[:, index, :]))
        return self.layer_probes
