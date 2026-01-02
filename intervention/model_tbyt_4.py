import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import types
from torch import nn, Tensor
from typing import Optional
import matplotlib.pyplot as plt

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

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
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size*2 + 1, config.block_size*2 + 1)).view(1,1,config.block_size*2 + 1, config.block_size*2 + 1))
        self.c_proj.NANOGPT_SCALE_INIT = True
        self.config = config
    def forward(self, x, layer_n=-1):
        #print('x shape is ', x.shape)
        B, T, C = x.size()
        #print(f'B: {B} T: {T} C:{C}')
        qkv = self.c_attn(x)
        #print(f'C: {C} self.n_embd: {self.n_embd}')
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        attn = q @ k.transpose(-1,-2) * 0.1 / (k.size(-1)) ** 0.5
        #print('attn dim is ', attn.shape)
        #print('bias is ', self.bias.shape)
        if layer_n == 0:
            #attn[:,:,40,22] += 40.0
            #attn[:,:,40,8] -= 40.0
            #attn[:,:,38,0] += 14.0
            #attn[:,:,38,21] += 1.0
            #attn[:,:,38,5] += 7.0
            #attn[:,:,38,21] += 1.0
            #attn[:,:,38,30] -= 0.1
            #attn[:,:,41,7] += 20.7
            #attn[:,:,44,5] += 4.0
            print('attn position 44,5 ', attn[:,:,44,5], ' position 44, 23 ', attn[:,:,44,23])
        attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
        self.raw_attn = attn.clone().detach().view(2*self.config.block_size + 1,2*self.config.block_size + 1)
        attn = F.softmax(attn, dim=-1)
        print('layer_n is ', layer_n)
        #if layer_n == 0:
        print('im setting the attn')
        self.attn = attn.clone().detach().view(2*self.config.block_size + 1,2*self.config.block_size + 1)
        

        if layer_n == 1:
            print('alaki')
            #attn[:,:,38,30] += 0.0
            #attn[:,:,38,5] += 2.0
        #if layer_n != -1:
            #print(f'attn scores of layer {layer_n} is {attn}')
            #import matplotlib.pyplot as plt
            #plt.matshow(attn.view(2*self.config.block_size + 1,2*self.config.block_size + 1).detach().numpy())
            #plt.matshow(attn.view(2*self.config.block_size + 1,2*self.config.block_size + 1)[48:, :32].detach().numpy())
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

    def forward(self, x, layer_n=-1):
        #print('im here!!!', x)
        if layer_n == 1:
            x = x + self.c_attn(self.ln_1(x), layer_n=layer_n)
            return x + self.c_fc(self.ln_2(x))
        else:
            x = x + self.c_attn(self.ln_1(x), layer_n=layer_n)
            return x + self.c_fc(self.ln_2(x))
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        #print('Im in GPT instructor')
        self.config = config
        self.n_layers = config.n_layers
        self.alpha = 100.0
        #print('i initialized n-layers')
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size + 1, config.n_embd),
            wpe = nn.Embedding(config.block_size * 4 + 1, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        #self.rope = RotaryPositionalEmbeddings(config.n_embd // config.n_heads, config.block_size * 4 + 1)
        #print('i initialized transformer')
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        #print('I have initialized all the variables in GPT instructor')
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
        device = idx.device
        pos = self.transformer.wpe(torch.arange(T).to(device))
        #print(f'idx device: {idx.device} wte device: {self.transformer.wte.weight.device}')
        
        x = self.transformer.wte(idx) #+ pos
        #x = self.rope(self.transformer.wte(idx))

        layer_n = 0
        for block in self.transformer.h:
            x = block(x, layer_n)
            layer_n += 1
        logits = self.lm_head(x)
        
        #v_loss_measure = torch.func.vmap(self.loss_measure)
        tensor1 = logits[:, self.config.block_size:T-1, :].contiguous().view(-1, logits.size(-1))
        tensor2 = idx[:, self.config.block_size + 1:].contiguous().view(-1)
        #print(f'tensor2: {tensor2.size()} tensor1: {tensor1.size()}')
        if flag == True:
            print(f'tensor1: {torch.softmax(tensor1, dim=-1)[torch.arange(tensor1.size(0)), tensor2]}')
        loss = F.cross_entropy(tensor1, tensor2)
        #F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, idx, topk, sampling_length):
        for round in range(sampling_length):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            _, indices_tmp = torch.topk(logits, 3, dim=-1)
            #print(f'logit indices for {idx[0, -1]} are {indices_tmp[0]}')
            vals, indices = torch.topk(logits, topk, -1, True)
            logits[logits < vals[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sampled_indices), dim=-1)
        return idx

    def proj_on_embedding(self, idx, index):
        B, T = idx.size()
        device = idx.device
        self.layer_probes = torch.zeros(B, self.config.n_layers + 1, self.config.vocab_size + 1).detach()
        #print(f'idx device: {idx.device} wte device: {self.transformer.wte.weight.device}')
        with torch.no_grad():
            pos = self.transformer.wpe(torch.arange(T).to(device))
            x = self.transformer.wte(idx) + pos
            #x = self.transformer.wte(idx)
            self.layer_probes[:, 0, :] = self.lm_head(x[:, index, :])
            for depth, block in enumerate(self.transformer.h, start=1):
                x = block(x)
                self.layer_probes[:, depth, :] = self.lm_head(x[:, index, :])
        return self.layer_probes

    def without_pos_embd(self, idx):
        B, T = idx.size()
        device = idx.device
        
        x = self.transformer.wte(idx)
        device = idx.device
        #x = self.transformer.wpe(torch.arange(T).to(device).view(1, T))

        print('T is ', T)
        #x[:,self.config.block_size, :] += self.transformer.wpe(torch.tensor(self.config.block_size))
        
        #x += self.transformer.wpe(torch.arange(T).to(device))
        #x = self.rope(self.transformer.wte(idx))

        layer_n = 0
        for block in self.transformer.h:
            layer_n += 1
            x = block(x, layer_n)
        logits = self.lm_head(x)
        
        tensor1 = logits[:, self.config.block_size:T-1, :].contiguous().view(-1, logits.size(-1))
        tensor2 = idx[:, self.config.block_size + 1:].contiguous().view(-1)
        loss = F.cross_entropy(tensor1, tensor2)
        return logits, loss


class GPTConfig():
    block_size: int = 32
    vocab_size: int = 128
    n_layers = 2
    n_heads = 1
    n_embd = 64

    def __init__(self, block_size=None, vocab_size=None):
        if block_size:
            self.block_size = block_size
        if vocab_size:
            self.vocab_size = vocab_size

    

class GPTIntervention:
    def __init__(self, gpt, idx):
        super().__init__()
        self.config = gpt.config
        self.gpt = gpt
        self.idx = idx
        _, _ = self.gpt(self.idx)
        self.attn = []
        self.attn.append(self.gpt.transformer.h[0].c_attn.attn)
        self.attn.append(self.gpt.transformer.h[1].c_attn.attn)
        self.raw_attn = []
        self.raw_attn.append(self.gpt.transformer.h[0].c_attn.raw_attn)
        self.raw_attn.append(self.gpt.transformer.h[1].c_attn.raw_attn)
        self.old_attention_forward = [None, None]

    def read_attention(self, attention_layer_num, location1, location2):
        return self.raw_attn[attention_layer_num][location1, location2]

    def check_if_still_works(self):
        logits, _ = self.gpt(self.idx)
        return torch.argmax(logits, dim=-1)[0, self.location].item() , self.idx[0, self.location + 1].item()

    def intervent_attention(self, attention_layer_num, location, unsorted_lb, unsorted_ub, unsorted_lb_num, unsorted_ub_num, unsorted_intensity_inc, sorted_lb, sorted_num, sorted_intensity_inc):
        self.location = location
        target_val = self.idx[0, location].item()
        next_number = self.idx[0, location + 1].item()
        unsorted_part = self.idx[0, :self.config.block_size]
        sorted_part = self.idx[0, self.config.block_size + 1:2*self.config.block_size + 1]
        
        # Pick unsorted_lb_num numbers: target_val - unsorted_lb <= x <= target_val
        unsorted_lb_mask = (unsorted_part >= target_val - unsorted_lb) & (unsorted_part <= target_val) & (unsorted_part != next_number)
        unsorted_lb_indices = torch.where(unsorted_lb_mask)[0]
        if len(unsorted_lb_indices) < unsorted_lb_num:
            raise Exception("Not enough numbers for unsorted_lb_num")
        unsorted_lb_selected = unsorted_lb_indices[torch.randperm(len(unsorted_lb_indices))[:unsorted_lb_num]]
        unsorted_lb_values = unsorted_part[unsorted_lb_selected]
        
        # Pick unsorted_ub_num numbers: target_val <= x <= target_val + unsorted_ub
        unsorted_ub_mask = (unsorted_part > target_val) & (unsorted_part <= target_val + unsorted_ub) & (unsorted_part != next_number)
        unsorted_ub_indices = torch.where(unsorted_ub_mask)[0]
        if len(unsorted_ub_indices) < unsorted_ub_num:
            raise Exception("Not enough numbers for unsorted_ub_num")
        unsorted_ub_selected = unsorted_ub_indices[torch.randperm(len(unsorted_ub_indices))[:unsorted_ub_num]] if len(unsorted_ub_indices) > 0 else torch.tensor([], dtype=torch.long)
        unsorted_ub_values = unsorted_part[unsorted_ub_selected] if len(unsorted_ub_selected) > 0 else torch.tensor([])
        
        # Pick sorted_num numbers: |x - target_val| <= sorted_lb
        sorted_mask = torch.abs(sorted_part - target_val) <= sorted_lb
        sorted_indices = torch.where(sorted_mask)[0]
        if len(sorted_indices) < sorted_num:
            raise Exception("Not enough numbers for sorted_num")
        sorted_selected = sorted_indices[torch.randperm(len(sorted_indices))[:sorted_num]]
        sorted_values = sorted_part[sorted_selected]
        sorted_actual_indices = sorted_selected + self.config.block_size + 1
        
        original_forward = self.gpt.transformer.h[attention_layer_num].c_attn.forward
        next_number_location = torch.where(self.idx[0, :self.config.block_size] == next_number)[0][0].item()
        main_attention_val = self.read_attention(attention_layer_num, location, next_number_location).item()

        def new_forward(self, x, layer_n=-1):
            B, T, C = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
            k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
            v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
            attn = q @ k.transpose(-1,-2) * 0.1 / (k.size(-1)) ** 0.5
            for index in unsorted_lb_selected:
                attn[:,:,location,index.item()] = main_attention_val + unsorted_intensity_inc
            for index in unsorted_ub_selected:
                #print(f'im adding unsorted_intensity_inc: {unsorted_intensity_inc} in unsorted_ub_selected: {index.item()}')
                
                attn[:,:,location,index.item()] = main_attention_val + unsorted_intensity_inc
            for index in sorted_actual_indices:
                attn[:,:,location,index.item()] = main_attention_val + sorted_intensity_inc

            attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
            #print('altered attention is ')

            
            attn = F.softmax(attn, dim=-1)
            #plt.plot(attn[0, 0, location, :].detach().numpy())
            #plt.savefig('plots_intervented_attention/altered_attention.png', dpi=150, bbox_inches='tight')
            self.new_attn = attn.view(2*self.config.block_size + 1,2*self.config.block_size + 1)

            print('next number is ', next_number)
            print('next number location is ', next_number_location)
            print('attn on next number location is ', attn[0,0,location, next_number_location].item()) 
            y = attn @ v
            y = y.transpose(1,2).contiguous().view(B,T,C)
            y = self.c_proj(y)
            #print('new forwarad is returning y with shape ', y.shape)
            return y
        
        attention_module = self.gpt.transformer.h[attention_layer_num].c_attn
        self.old_attention_forward[attention_layer_num] = attention_module.forward
        attention_module.forward = types.MethodType(new_forward, attention_module)
        return self.gpt, ((unsorted_lb_selected, unsorted_lb_values), (unsorted_ub_selected, unsorted_ub_values), (sorted_actual_indices, sorted_values))

    def revert_attention(self, attention_layer_num):
        if self.old_attention_forward[attention_layer_num] is None:
            raise Exception("No old attention forward found")
        attention_module = self.gpt.transformer.h[attention_layer_num].c_attn
        attention_module.forward = self.old_attention_forward[attention_layer_num]
        return self.gpt
    
    def get_attention_matrix(self, attention_layer_num):
        return self.gpt.transformer.h[attention_layer_num].c_attn.attn