"""
Model definition for analysis/plotting of grid-run checkpoints.
Same architecture as model_tbyt_train.py but with:
  - Attention weight storage (raw_attn, attn) in CasualSelfAttention
  - GPTIntervention class for attention intervention experiments
Weight-compatible with model_tbyt_train.py checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import types


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
        seq_len = config.block_size * 2 + 1
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
        self.raw_attn = attn.clone().detach().view(T, T)
        attn = F.softmax(attn, dim=-1)
        self.attn = attn.clone().detach().view(T, T)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, layer_n=-1):
        x = x + self.c_attn(self.ln_1(x), layer_n=layer_n)
        return x + self.c_fc(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.alpha = 100.0
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
            wpe=nn.Embedding(config.block_size * 4 + 1, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd),
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
        for layer_n, block in enumerate(self.transformer.h):
            x = block(x, layer_n)
        if self.config.with_layer_norm:
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        tensor1 = logits[:, self.config.block_size:T - 1, :].contiguous().view(-1, logits.size(-1))
        tensor2 = idx[:, self.config.block_size + 1:].contiguous().view(-1)
        loss = F.cross_entropy(tensor1, tensor2)
        return logits, loss


class GPTConfig:
    block_size: int = 32
    vocab_size: int = 128
    n_layers = 2
    n_heads = 1
    n_embd = 64
    with_layer_norm: bool = False

    def __init__(self, block_size=None, vocab_size=None, with_layer_norm=False):
        if block_size:
            self.block_size = block_size
        if vocab_size:
            self.vocab_size = vocab_size
        self.with_layer_norm = with_layer_norm


class GPTIntervention:
    def __init__(self, gpt, idx):
        super().__init__()
        self.config = gpt.config
        self.gpt = gpt
        self.idx = idx
        _, _ = self.gpt(self.idx)
        self.attn = [self.gpt.transformer.h[i].c_attn.attn for i in range(self.config.n_layers)]
        self.raw_attn = [self.gpt.transformer.h[i].c_attn.raw_attn for i in range(self.config.n_layers)]
        self.old_attention_forward = [None] * self.config.n_layers

    def read_attention(self, layer, loc1, loc2):
        return self.raw_attn[layer][loc1, loc2]

    def check_if_still_works(self):
        logits, _ = self.gpt(self.idx)
        return (torch.argmax(logits, dim=-1)[0, self.location].item(),
                self.idx[0, self.location + 1].item())

    def intervent_attention(self, attention_layer_num, location,
                            unsorted_lb, unsorted_ub,
                            unsorted_lb_num, unsorted_ub_num,
                            unsorted_intensity_inc,
                            sorted_lb, sorted_num, sorted_intensity_inc):
        self.location = location
        target_val = self.idx[0, location].item()
        next_number = self.idx[0, location + 1].item()
        unsorted_part = self.idx[0, :self.config.block_size]
        sorted_part = self.idx[0, self.config.block_size + 1:2 * self.config.block_size + 1]

        unsorted_lb_mask = ((unsorted_part >= target_val - unsorted_lb)
                            & (unsorted_part <= target_val)
                            & (unsorted_part != next_number))
        unsorted_lb_indices = torch.where(unsorted_lb_mask)[0]
        if len(unsorted_lb_indices) < unsorted_lb_num:
            raise Exception("Not enough numbers for unsorted_lb_num")
        unsorted_lb_selected = unsorted_lb_indices[torch.randperm(len(unsorted_lb_indices))[:unsorted_lb_num]]

        unsorted_ub_mask = ((unsorted_part > target_val)
                            & (unsorted_part <= target_val + unsorted_ub)
                            & (unsorted_part != next_number))
        unsorted_ub_indices = torch.where(unsorted_ub_mask)[0]
        if len(unsorted_ub_indices) < unsorted_ub_num:
            raise Exception("Not enough numbers for unsorted_ub_num")
        unsorted_ub_selected = (unsorted_ub_indices[torch.randperm(len(unsorted_ub_indices))[:unsorted_ub_num]]
                                if len(unsorted_ub_indices) > 0 else torch.tensor([], dtype=torch.long))

        sorted_mask = torch.abs(sorted_part - target_val) <= sorted_lb
        sorted_indices = torch.where(sorted_mask)[0]
        if len(sorted_indices) < sorted_num:
            raise Exception("Not enough numbers for sorted_num")
        sorted_selected = sorted_indices[torch.randperm(len(sorted_indices))[:sorted_num]]
        sorted_actual_indices = sorted_selected + self.config.block_size + 1

        next_number_location = torch.where(self.idx[0, :self.config.block_size] == next_number)[0][0].item()
        main_attention_val = self.read_attention(attention_layer_num, location, next_number_location).item()
        config = self.config

        def new_forward(self_attn, x, layer_n=-1):
            B, T, C = x.size()
            qkv = self_attn.c_attn(x)
            q, k, v = qkv.split(self_attn.n_embd, dim=2)
            q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
            k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
            v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
            attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
            for index in unsorted_lb_selected:
                attn[:, :, location, index.item()] = main_attention_val + unsorted_intensity_inc
            for index in unsorted_ub_selected:
                attn[:, :, location, index.item()] = main_attention_val + unsorted_intensity_inc
            for index in sorted_actual_indices:
                attn[:, :, location, index.item()] = main_attention_val + sorted_intensity_inc
            attn = attn.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self_attn.c_proj(y)
            return y

        attention_module = self.gpt.transformer.h[attention_layer_num].c_attn
        self.old_attention_forward[attention_layer_num] = attention_module.forward
        attention_module.forward = types.MethodType(new_forward, attention_module)

    def revert_attention(self, attention_layer_num):
        if self.old_attention_forward[attention_layer_num] is None:
            raise Exception("No old attention forward found")
        self.gpt.transformer.h[attention_layer_num].c_attn.forward = self.old_attention_forward[attention_layer_num]
