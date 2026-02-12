"""
GPTIntervention class for models loaded via load_checkpoint (model_tbyt_3_withnormalization.py)
"""
import torch
import torch.nn.functional as F
import types
import math


class GPTInterventionNormalized:
    """
    Intervention class for models using model_tbyt_3_withnormalization.py architecture.
    
    Key differences from original GPTIntervention:
    - Accesses attention via h[i].attn instead of h[i].c_attn
    - Uses attn_weights instead of attn/raw_attn
    - Compatible with CausalSelfAttention that uses scaled_dot_product_attention
    """
    
    def __init__(self, gpt, idx):
        self.config = gpt.config
        self.gpt = gpt
        self.idx = idx
        
        # Run forward pass to populate attention weights
        _, _ = self.gpt(self.idx)
        
        # Access attention weights (stored in CausalSelfAttention.attn_weights)
        self.attn = []
        self.raw_attn = []
        for i in range(self.config.n_layers):
            attn_module = self.gpt.transformer.h[i].attn
            if hasattr(attn_module, 'attn_weights') and attn_module.attn_weights is not None:
                self.attn.append(attn_module.attn_weights.clone())
                # For normalized model, we store softmax attention as "raw" too
                # (the original model stored pre-softmax as raw_attn)
                self.raw_attn.append(attn_module.attn_weights.clone())
            else:
                raise ValueError(f"Attention weights not found in layer {i}. "
                                "Make sure model_tbyt_3_withnormalization.py stores attn_weights.")
        
        self.old_attention_forward = [None] * self.config.n_layers
        self.location = None

    def read_attention(self, attention_layer_num, location1, location2):
        return self.raw_attn[attention_layer_num][location1, location2]

    def check_if_still_works(self):
        logits, _ = self.gpt(self.idx)
        return torch.argmax(logits, dim=-1)[0, self.location].item(), self.idx[0, self.location + 1].item()

    def intervent_attention(self, attention_layer_num, location, unsorted_lb, unsorted_ub, 
                           unsorted_lb_num, unsorted_ub_num, unsorted_intensity_inc, 
                           sorted_lb, sorted_num, sorted_intensity_inc):
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
        
        print(f'unsorted_lb_selected is ', unsorted_lb_selected)
        print(f'unsorted_lb_values is ', unsorted_lb_values)
        
        # Pick unsorted_ub_num numbers: target_val < x <= target_val + unsorted_ub
        unsorted_ub_mask = (unsorted_part > target_val) & (unsorted_part <= target_val + unsorted_ub) & (unsorted_part != next_number)
        unsorted_ub_indices = torch.where(unsorted_ub_mask)[0]
        if len(unsorted_ub_indices) < unsorted_ub_num:
            raise Exception("Not enough numbers for unsorted_ub_num")
        unsorted_ub_selected = unsorted_ub_indices[torch.randperm(len(unsorted_ub_indices))[:unsorted_ub_num]] if len(unsorted_ub_indices) > 0 else torch.tensor([], dtype=torch.long)
        unsorted_ub_values = unsorted_part[unsorted_ub_selected] if len(unsorted_ub_selected) > 0 else torch.tensor([])
        
        print(f'unsorted_ub_selected is ', unsorted_ub_selected)
        print(f'unsorted_ub_values is ', unsorted_ub_values)
        
        # Pick sorted_num numbers: |x - target_val| <= sorted_lb
        sorted_mask = torch.abs(sorted_part - target_val) <= sorted_lb
        sorted_indices = torch.where(sorted_mask)[0]
        if len(sorted_indices) < sorted_num:
            raise Exception("Not enough numbers for sorted_num")
        sorted_selected = sorted_indices[torch.randperm(len(sorted_indices))[:sorted_num]]
        sorted_values = sorted_part[sorted_selected]
        sorted_actual_indices = sorted_selected + self.config.block_size + 1
        
        print(f'sorted_actual_indices is ', sorted_actual_indices)
        print(f'sorted_values is ', sorted_values)
        
        next_number_location = torch.where(self.idx[0, :self.config.block_size] == next_number)[0][0].item()
        main_attention_val = self.read_attention(attention_layer_num, location, next_number_location).item()
        
        # Get the attention module
        attention_module = self.gpt.transformer.h[attention_layer_num].attn
        head_dim = attention_module.head_dim
        n_heads = attention_module.n_heads
        n_embd = attention_module.n_embd
        device = self.idx.device
        
        def new_forward(self, x):
            B, T, C = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Manual attention computation
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            
            # Apply interventions (in pre-softmax space)
            for index in unsorted_lb_selected:
                attn[:, :, location, index.item()] += unsorted_intensity_inc
            for index in unsorted_ub_selected:
                attn[:, :, location, index.item()] += unsorted_intensity_inc
            for index in sorted_actual_indices:
                attn[:, :, location, index.item()] += sorted_intensity_inc
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            
            # Store attention weights
            self.attn_weights = attn[0, 0].detach()
            
            print('next number is ', next_number)
            print('next number location is ', next_number_location)
            print('attn on next number location is ', attn[0, 0, location, next_number_location].item())
            
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            return y
        
        self.old_attention_forward[attention_layer_num] = attention_module.forward
        attention_module.forward = types.MethodType(new_forward, attention_module)
        
        return self.gpt, ((unsorted_lb_selected, unsorted_lb_values), 
                         (unsorted_ub_selected, unsorted_ub_values), 
                         (sorted_actual_indices, sorted_values))

    def revert_attention(self, attention_layer_num):
        if self.old_attention_forward[attention_layer_num] is None:
            return self.gpt
        attention_module = self.gpt.transformer.h[attention_layer_num].attn
        attention_module.forward = self.old_attention_forward[attention_layer_num]
        self.old_attention_forward[attention_layer_num] = None
        return self.gpt
    
    def get_attention_matrix(self, attention_layer_num):
        return self.gpt.transformer.h[attention_layer_num].attn.attn_weights
