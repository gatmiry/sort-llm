## this one is simlar to ddmodel except that 
# each layer is trained on pairs of sequences and their once flipped version for a random pair

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
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
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        #print(f'C: {C} self.n_embd: {self.n_embd}')
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        attn = q @ k.transpose(-1,-2) / (k.size(-1)) ** -0.5
        #attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        #print('im in block instructor')
        super().__init__()
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        #print('i initialized everying in block')

    def forward(self, x):
        x = x + self.c_attn(self.ln_1(x))
        return x + self.c_fc(self.ln_2(x))
    
class DDGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print('Im in GPT instructor')
        self.n = config.vocab_size
        self.n_layers = config.n_layers
        self.alpha = 100.0
        print('i initialized n-layers')
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        print('i initialized transformer')
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        print('I have initialized all the variables in GPT instructor')
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

    ## have changed this one
    def forward(self, targets=None, gen=False):
        device = targets[0].device
        #print(f'idx device: {idx.device} wte device: {self.transformer.wte.weight.device}')
    

        count = 0
        loss = 0.0
        if not gen:
            B, T = targets[0].size()
            idx = torch.arange(0, T, dtype=torch.long, device=device)
            for block in self.transformer.h:
                x = self.transformer.wte(targets[count]) + self.transformer.wpe(idx)
                x = block(x)
                logits = self.lm_head(x)
                loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets[count + 1].view(-1))
                count += 1
        else:
            B, T = targets.size()
            idx = torch.arange(0, T, dtype=torch.long, device=device)
            x = self.transformer.wte(targets) + self.transformer.wpe(idx)
            for block in self.transformer.h:
                x = block(x)
            logits = self.lm_head(x)
            loss = None    
        #logits = self.lm_head(x)
        #loss += F.cross_entropy(logits.view(-1, logits.size(-1), targets[count].view(-1)))
        
        return logits, loss
    
    
    def generate(self, idx):
        logits, _ = self(idx, True)
        indices = torch.argmax(logits, dim=-1)
        return indices
    
class PerturberBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tokens):
        B, T = tokens.size()
        #idx1 = torch.randint(0, T, (B,))
        #idx2 = torch.randint(0, T, (B,))
        idx1, idx2 = torch.tensor([0]), torch.tensor([1])
        idx1 = idx1.repeat(B)
        idx2 = idx2.repeat(B)
        tmp = tokens[np.arange(B), idx1]
        tokens = tokens.clone()
        tokens[np.arange(B), idx1] = tokens[np.arange(B), idx2]
        tokens[np.arange(B), idx2] = tmp
        #print('tokens: ', tokens, ' tmp: ', tmp)
        return tokens

from collections import deque
class Perturber(nn.Module):
    def __init__(self, n_layers, swap_count):
        super().__init__()
        self.swap_count = swap_count
        self.n_layers = n_layers
        self.perturbers = nn.ModuleList([PerturberBlock() for _ in range(n_layers)])
    def forward(self, x):
        pseqs = deque([x])
        for i in range(len(self.perturbers)):
            for sp in range(self.swap_count):
                x = self.perturbers[i](x)
            pseqs.appendleft(x)
        return pseqs

class GPTConfig():
    block_size: int = 8
    vocab_size: int = 128
    n_layers = 1
    n_heads = 8
    n_embd = 1024

    def __init__(self, block_size=None, vocab_size=None):
        if block_size:
            self.block_size = block_size
        if vocab_size:
            self.vocab_size = vocab_size

    