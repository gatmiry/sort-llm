import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from typing import Optional

import torch

from torch import nn, Tensor


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
    def forward(self, x, layer_n=-1):
        #if layer_n == 0:
        #    return self.fc_2(self.fc_1(x))
        #else:
        #    return self.fc_2(self.gelu(self.fc_1(x)))
        return self.fc_2(self.gelu(self.fc_1(x)))

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size*2 + 1, config.block_size*2 + 1)).view(1,1,config.block_size*2 + 1, config.block_size*2 + 1))
        print('bias diag is ', self.bias[0,0,0,:])
        self.c_proj.NANOGPT_SCALE_INIT = True
        self.config = config

    def forward(self, x, layer_n=-1, return_attention=False, word_embeddings=None, idx=None):
        print('x shape is ', x.shape)
        B, T, C = x.size()
        #print(f'B: {B} T: {T} C:{C}')
        qkv = self.c_attn(x)
        #print('thissss is ', word_embeddings)
        if word_embeddings is not None:
            _, _, word_v = self.c_attn(word_embeddings).split(self.n_embd, dim=2)  
        #print(f'C: {C} self.n_embd: {self.n_embd}')
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        attn = q @ k.transpose(-1,-2) * 0.1 / (k.size(-1)) ** 0.5
        #print('attn dim is ', attn.shape)
        #print('bias is ', self.bias.shape)
        print('attn dim is ', attn.shape)
        attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))

        ## cutting off the tail of the scores
        if idx is not None:
            cond = idx.unsqueeze(1) <= idx.unsqueeze(0)
            #attn = torch.where(cond, attn, 0.0) 


        ## manually change the attention in positions 9, 22, 27 --> position 17 has the right number which is 20
        #print('attn location 9 is ', attn[0,0,35,9])
        #print('attn location 17 is ', attn[0,0,35,17])
        if layer_n == 0:
            attn[0,0,35,5] = attn[0,0,35,5] - 5.0
            attn[0,0,35,4] = attn[0,0,35,4] + 20.0
            #attn[0,0,35,3] = attn[0,0,35,3] + 16.0
            attn[0,0,35,9] = attn[0,0,35,9] + 16.0
            attn[0,0,35,22] = attn[0,0,35,22] - 20.0
            attn[0,0,35,27] = attn[0,0,35,27] - 6.0
            attn[0,0,35,17] = attn[0,0,35,17] - 6.0
            attn[0,0,35,35] = attn[0,0,35,35] + 14.7#8.2#new sweet spot is 3.8 #+ 1.5 #this is the sweet spot # seems like it ranges monotonically between 16 and 107 as you vary this score
            #print('entry 22 is ', idx[0,22])
            #print('entry 9 is ', idx[0,9])
        attn = F.softmax(attn, dim=-1)
        if layer_n == 0:
            self.attn = attn
        #print('attn[:,:,33,:] is ', attn[:,:,33,:])
        if layer_n != -1:
            #print(f'attn scores of layer {layer_n} is {attn}')
            import matplotlib.pyplot as plt
            #print('im here!!! ', attn.size())
            #print('attn max from index 32 on are ', idx[0,torch.argmax(attn[0,0,32:,:], dim=1)])
            #mat = plt.matshow(attn.view(2*self.config.block_size + 1, 2*self.config.block_size + 1).detach().numpy())
            #plt.colorbar(mat, label='intensity')
            #plt.show()
            
        y = attn @ v 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        #if word_embeddings is not None:
        #    print('im doing the additional work!!')
        #    y = y - attn.view(B, T, T) @ word_v + word_v
        y = self.c_proj(y)
        if return_attention:
            return attn
        else:
            return y

class Block(nn.Module):
    def __init__(self, config, position=-1, use_attention=True):
        #print('im in block instructor')
        super().__init__()
        self.position = position
        self.normval = None
        self.meanval = None
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.config = config
        self.use_attention = use_attention
        #print('i initialized everying in block')

    def forward(self, x, layer_n=-1, midvec=None, midvec2=None, pos_embeddings=None, word_embeddings=None, idx=None):
        #print('im here!!!', x)
        B, T, C = x.size()
        import matplotlib.pyplot as plt
        first_in_batch = []
        #for position in range(65):
        
        
        #first_in_batch.append(2000 * Vh[0,0].item())
        
        #print(f'singular values of layer {layer_n} are ')
        #print('S dims are ', S.shape)
        #plt.figure(1)
        #plt.plot(S.detach().numpy())
        #plt.show()
        #print(f'first singular direction of layer {layer_n} is ', )
        #plt.plot(200* U[:,0].detach().numpy())
        #plt.figure(2)
        #plt.plot(first_in_batch)
        #plt.show()
        U, S, Vh = torch.linalg.svd(self.c_attn(self.ln_1(x))[:,self.position,:])
        #print('S is ')
        #plt.plot(S.detach().to('cpu').numpy())
        #plt.show()
        #print('norm without residual ', torch.norm(self.c_attn(self.ln_1(x))))
        #print('residual part norm ', torch.norm(x))
        #print('idxxxx is ', idx)
        if midvec == None or midvec2 == None or pos_embeddings == None or word_embeddings == None:
            #print('word embeddings are ', word_embeddings)
            print('don have to be here!')
            x = x + self.c_attn(self.ln_1(x), layer_n=layer_n)
            #x = x
        else:
            x = x + self.c_attn(self.ln_1(x), layer_n=layer_n, word_embeddings=word_embeddings, idx=idx)
            #if layer_n == 1:
            #    x = x
            #else:
            #    x = x + self.c_attn(self.ln_1(x), layer_n=layer_n, word_embeddings=word_embeddings, idx=idx)
        #    qweights, kweights, vweights = self.c_attn.c_attn.weight.split(self.config.n_embd, dim=0)
        #    attn = self.c_attn(self.ln_1(x), layer_n=layer_n, return_attention=True)
        #    print('shape is ', (attn @ self.ln_1(x)).transpose(1,2).shape)
            #print('critical point is ', (attn @ self.ln_1(pos_embeddings[:65, :]).repeat(B, 1, 1, 1)).shape)
        #    x = x + self.c_attn.c_proj((self.ln_1(x)/2 + (attn @ self.ln_1(pos_embeddings[:65, :]).repeat(B, 1, 1, 1)).transpose(1,2).view(B, T, C)/2) @ vweights.t())#self.c_attn.c_proj((self.ln_1(x) + 0.2* self.ln_1(midvec + midvec2)) @ vweights.t())#self.c_attn(self.ln_1(x), layer_n=layer_n)

        
        self.Vh = Vh
        self.normval = torch.sqrt(1e-5 + torch.var(x, dim=-1, correction=0, keepdim=True).detach())
        self.meanval = torch.mean(x, dim=-1).detach().unsqueeze(-1)
        #if layer_n == 0:
        #    return x    
        #else:
        return x + self.c_fc(self.ln_2(x), layer_n = layer_n)
        #if layer_n == 0:
        #    return x + self.c_fc(self.ln_2(x))
        #if layer_n == 1:
        #    return x 
        
    
class GPT(nn.Module):
    def __init__(self, config, position=-1):
        super().__init__()
        print('Im in GPT instructor')
        self.config = config
        self.n_layers = config.n_layers
        self.alpha = 100.0
        self.norms = None
        self.means = None
        self.position = position
        print('i initialized n-layers')
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size + 1, config.n_embd),
            wpe = nn.Embedding(config.block_size * 4 + 1, config.n_embd),
            h = nn.ModuleList([Block(config, self.position) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        #self.rope = RotaryPositionalEmbeddings(config.n_embd // config.n_heads, config.block_size * 4 + 1)
        print('i initialized transformer')
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        print('I have initialized all the variables in GPT instructor')
        self.apply(self._init_weights)
        

    def set_position(self, position):
        self.position = position
        for block in self.transformer.h:
            block.position = position    
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
        #import matplotlib.pyplot as plt
        #print('numbers are ')
        #plt.plot(idx[:,34])
        B, T = idx.size()
        device = idx.device
        pos = self.transformer.wpe(torch.arange(T).to(device))
        pos_sizes = (pos @ pos.t()).detach().to('cpu').numpy()
        import matplotlib.pyplot as plt
        #print('pos_sizes matshow:')
        #mat = plt.matshow(pos_sizes[32:, 32:])
        #plt.colorbar(mat, label='intensity')
        #plt.show()
        
        self.norms = torch.empty((self.config.n_layers, B, T, 1))
        self.means = torch.empty((self.config.n_layers, B, T, 1))
        if self.config.without_pos:
            x = self.transformer.wte(idx)
        else:
            x = self.transformer.wte(idx) + pos
        #x = self.rope(self.transformer.wte(idx))

        layer_n = 0
        self.Vhs = []
        for block in self.transformer.h:
            #print('avali is ', torch.var(x, dim=-1, correction=0, keepdim=True).detach().size())
            self.norms[layer_n,:,:] = torch.sqrt(1e-5 + torch.var(x, dim=-1, correction=0, keepdim=True).detach())
            self.means[layer_n,:,:] = torch.mean(x, dim=-1).detach().unsqueeze(-1)
            x = block(x, layer_n, self.transformer.wpe.weight[32,:], self.transformer.wte.weight[128,:], self.transformer.wpe.weight, self.transformer.wte(idx), idx)
            layer_n += 1
            ###
            self.Vhs.append(block.Vh)
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
            self.layer_probes[:, 0, :] = self.lm_head(x[:, index, :])
            for depth, block in enumerate(self.transformer.h, start=1):
                x = block(x)
                self.layer_probes[:, depth, :] = self.lm_head(x[:, index, :])
        return self.layer_probes

    def without_pos_embd(self, idx):
        B, T = idx.size()
        device = idx.device
        
        with torch.no_grad():
            #pos = self.transformer.wpe(torch.arange(T).to(device))
            #pos = self.transformer.wpe(torch.tensor([1]).repeat(T).to(device))
            x = self.transformer.wte(idx)
            #x = x + self.transformer.wpe(torch.arange(T).to(device).view(1, T))
            pos_indices = torch.tensor([0])
            pos_indices = torch.arange(T).to(device).view(1,T)
            pos_indices[:,33:64] = 64
            #pos_indices = pos_indices.repeat(1, T).to(device)
            pos = self.transformer.wpe(pos_indices)
            #pos[:,self.config.block_size, :] = self.transformer.wpe(torch.tensor(self.config.block_size))
            x = x + pos
            
            #x = self.transformer.w

            #x[:,self.config.block_size, :] += self.transformer.wpe(torch.tensor(self.config.block_size))
            
            #x += self.transformer.wpe(torch.arange(T).to(device))
            #x = self.rope(self.transformer.wte(idx))

            ## first normalize the token embeddings with the correct norm including the positional embeddings
            layer_n = 0
            print('x size is ', x.size())
            for _ in self.transformer.h:
                x0 = x
                x = x - self.means[layer_n, :, :]
                x = x / self.norms[layer_n, :, :]
                x = x * self.transformer.h[layer_n].ln_1.weight + self.transformer.h[layer_n].ln_1.bias
                x = x0 + self.transformer.h[layer_n].c_attn(x, layer_n)
                ## next we apply MLP
                x1 = x
                x = x - self.transformer.h[layer_n].meanval
                x = x / self.transformer.h[layer_n].normval
                x = x * self.transformer.h[layer_n].ln_2.weight + self.transformer.h[layer_n].ln_2.bias
                x = x1 + self.transformer.h[layer_n].c_fc(x)
                #x = block(x, layer_n)
                layer_n += 1

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
    without_pos = False

    def __init__(self, block_size=None, vocab_size=None):
        if block_size:
            self.block_size = block_size
        if vocab_size:
            self.vocab_size = vocab_size

    