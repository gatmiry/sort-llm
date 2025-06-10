import torch.nn as nn
import torch.nn.functional as F
import torch
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
        print('im in block instructor')
        super().__init__()
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        print('i initialized everying in block')

    def forward(self, x):
        x = x + self.c_attn(self.ln_1(x))
        return x + self.c_fc(self.ln_2(x))
    
class GPT(nn.Module):
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        #print(f'idx device: {idx.device} wte device: {self.transformer.wte.weight.device}')
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(x)
        
        #v_loss_measure = torch.func.vmap(self.loss_measure)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            #loss = self.myloss(logits, targets)
        #F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def myloss(self, logits, targets):
        B, T, C = logits.size()
        ## building up the new loss
        #probs = torch.softmax(logits)
        #log_normalization = torch.log(torch.sum(torch.exp(logits), dim=-1))
        log_normalization = torch.logsumexp(logits, dim=-1)
        tmp = -(logits - log_normalization.unsqueeze(-1))
        tmp1 = torch.stack([torch.roll(tmp, i, 1) * self.alpha for i in range(T)])
        tmp2 = torch.stack([torch.roll(tmp, -i, 1) + i*self.alpha for i in range(T)])
        #print(f'size tmp1: {tmp1.size()}')
        tmp = torch.cat((tmp1, tmp2), dim=0)
        vals, indices = torch.min(tmp, dim=0)
        loss = torch.gather(vals, dim=2, index=targets.unsqueeze(-1)).mean()
        return loss


    ## this one is outdated
    def loss_measure(self, seq1, seq2):
        assert len(seq1) == self.n, 'two sequences provided in loss_measure should have equal length'
        total = 0
        mydict = {}
        sol = {}
        # build sol
        for i in range(len(seq2)):
            sol[seq2[i]] = i

        for i in range(len(seq1)):
            if seq1[i] not in sol:
                total += self.n
            elif seq1[i] not in mydict:
                mydict[seq1[i]] = abs(sol[seq1[i]] - i)
            else:
                mydict[seq1[i]] = min(mydict[seq1[i]], abs(sol[seq1[i]] - i))
        for num in sol:
            if num not in mydict:
                total += self.n
            else:
                total += mydict[num]
        return total / self.n
    def generate_onepass(self, idx):
        logits, _ = self(idx)
        indices = torch.argmax(logits, dim=-1)
        return indices

    def generate_tbyt(self, idx, topk, sampling_length):
        for round in range(sampling_length):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            vals, indices = torch.topk(logits, topk, dim=-1)
            logits[logits < vals[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sampled_indices), dim=-1)
        return idx
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

    