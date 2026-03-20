import sys
import os
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from model_final import GPT, GPTConfig
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GPTIntervention:
    def __init__(self, gpt, idx, block_size):
        self.config = gpt.config
        self.gpt = gpt
        self.idx = idx
        self.block_size = block_size
        _, _ = self.gpt(self.idx, block_size=block_size, return_full_logits=True)
        self.attn = [self.gpt.transformer.h[i].attn.stored_attn for i in range(self.config.n_layers)]
        self.raw_attn = [self.gpt.transformer.h[i].attn.raw_attn for i in range(self.config.n_layers)]
        self.old_attention_forward = [None] * self.config.n_layers

    def read_attention(self, layer, loc1, loc2):
        return self.raw_attn[layer][loc1, loc2]

    def check_if_still_works(self):
        logits, _ = self.gpt(self.idx, block_size=self.block_size, return_full_logits=True)
        return torch.argmax(logits, dim=-1)[0, self.location].item(), self.idx[0, self.location + 1].item()

    def intervent_attention(self, attention_layer_num, location, unsorted_lb, unsorted_ub,
                            unsorted_lb_num, unsorted_ub_num, unsorted_intensity_inc,
                            sorted_lb, sorted_num, sorted_intensity_inc):
        self.location = location
        bs = self.block_size
        target_val = self.idx[0, location].item()
        next_number = self.idx[0, location + 1].item()
        unsorted_part = self.idx[0, :bs]
        sorted_part = self.idx[0, bs + 1:2 * bs + 1]

        unsorted_lb_mask = (unsorted_part >= target_val - unsorted_lb) & (unsorted_part <= target_val) & (unsorted_part != next_number)
        unsorted_lb_indices = torch.where(unsorted_lb_mask)[0]
        if len(unsorted_lb_indices) < unsorted_lb_num:
            raise Exception("Not enough numbers for unsorted_lb_num")
        unsorted_lb_selected = unsorted_lb_indices[torch.randperm(len(unsorted_lb_indices))[:unsorted_lb_num]]

        unsorted_ub_mask = (unsorted_part > target_val) & (unsorted_part <= target_val + unsorted_ub) & (unsorted_part != next_number)
        unsorted_ub_indices = torch.where(unsorted_ub_mask)[0]
        if len(unsorted_ub_indices) < unsorted_ub_num:
            raise Exception("Not enough numbers for unsorted_ub_num")
        unsorted_ub_selected = unsorted_ub_indices[torch.randperm(len(unsorted_ub_indices))[:unsorted_ub_num]] if len(unsorted_ub_indices) > 0 else torch.tensor([], dtype=torch.long)

        sorted_mask = torch.abs(sorted_part - target_val) <= sorted_lb
        sorted_indices = torch.where(sorted_mask)[0]
        if len(sorted_indices) < sorted_num:
            raise Exception("Not enough numbers for sorted_num")
        sorted_selected = sorted_indices[torch.randperm(len(sorted_indices))[:sorted_num]]
        sorted_actual_indices = sorted_selected + bs + 1

        next_number_location = torch.where(self.idx[0, :bs] == next_number)[0][0].item()
        main_attention_val = self.read_attention(attention_layer_num, location, next_number_location).item()

        def new_forward(self_attn, x):
            B, T, C = x.size()
            qkv = self_attn.c_attn(x)
            q, k, v = qkv.split(self_attn.n_embd, dim=2)
            q = q.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
            k = k.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
            v = v.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
            attn = q @ k.transpose(-1, -2) / (self_attn.head_dim ** 0.5)

            for index in unsorted_lb_selected:
                attn[:, :, location, index.item()] = main_attention_val + unsorted_intensity_inc
            for index in unsorted_ub_selected:
                attn[:, :, location, index.item()] = main_attention_val + unsorted_intensity_inc
            for index in sorted_actual_indices:
                attn[:, :, location, index.item()] = main_attention_val + sorted_intensity_inc

            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
            self_attn.raw_attn = attn.clone().detach().squeeze(0).squeeze(0)
            attn = F.softmax(attn, dim=-1)
            self_attn.stored_attn = attn.clone().detach().squeeze(0).squeeze(0)
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self_attn.c_proj(y)
            return y

        attn_module = self.gpt.transformer.h[attention_layer_num].attn
        self.old_attention_forward[attention_layer_num] = attn_module.forward
        attn_module.forward = types.MethodType(new_forward, attn_module)

    def revert_attention(self, attention_layer_num):
        if self.old_attention_forward[attention_layer_num] is None:
            raise Exception("No old attention forward found")
        attn_module = self.gpt.transformer.h[attention_layer_num].attn
        attn_module.forward = self.old_attention_forward[attention_layer_num]


device = 'cuda'
num_rounds = 1000
FN = 200
intensity_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

block_size = 32
vocab_size = 128

nathan_checkpoints = {
    'without': 'sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln0_wd0p0_seed1337__final.pt',
    'with': 'sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln1_wd0p0_seed1337__final.pt',
}

def get_batch():
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def run_experiment(wlnorm):
    ckpt_path = os.path.join(os.path.dirname(__file__), f'../../saved_models/nathanmodels/{nathan_checkpoints[wlnorm]}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_config'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    location = block_size + 5

    results = {}
    for intensity in intensity_values:
        attempts = []
        for _ in range(num_rounds):
            if len(attempts) >= FN:
                break
            idx = get_batch().to(device)
            try:
                intervention = GPTIntervention(model, idx, block_size)
                intervention.intervent_attention(
                    attention_layer_num=1, location=location,
                    unsorted_lb=5, unsorted_ub=5, unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intensity, sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0
                )
                new_num, next_num = intervention.check_if_still_works()
                attempts.append(new_num == next_num)
                intervention.revert_attention(1)
            except:
                continue
        first_fn = attempts[:FN]
        results[intensity] = sum(first_fn) / len(first_fn) if first_fn else 0
        print(f'[{wlnorm} LN] Intensity: {intensity:+.2f}, Success: {results[intensity]:.4f} ({sum(first_fn)}/{len(first_fn)})')
    return results


print("=" * 50)
print("Model WITHOUT final layer normalization (fln0)")
print("=" * 50)
results_without = run_experiment('without')

print("\n" + "=" * 50)
print("Model WITH final layer normalization (fln1)")
print("=" * 50)
results_with = run_experiment('with')

plt.figure(figsize=(3.5, 2.8))
intensities = list(results_without.keys())
plt.plot(intensities, list(results_without.values()), marker='o', linewidth=1.5, markersize=5,
         label='Without LayerNorm', color='#6a3d9a')
plt.plot(intensities, list(results_with.values()), marker='s', linewidth=1.5, markersize=5,
         label='With LayerNorm', color='#e6850e')
plt.xlabel('Intervention Intensity', fontsize=9)
plt.ylabel('Success Probability', fontsize=9)
plt.title('Robustness to Attention Intervention', fontsize=10)
plt.legend(fontsize=7, loc='lower left')
plt.grid(True, alpha=0.3)
plt.xticks(intensities[::2], fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(0, 1.05)
plt.tight_layout()

output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, 'compare_intensity_layer1.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'\nPlot saved to {output_path}')
