# SortGPT Agent Context

This file provides the technical context needed for a new agent to continue
the mechanistic interpretability work on SortGPT models. For a human-readable
summary of findings, see `REPORT.md`.

---

## 1. Repository Structure

```
/mnt/task_runtime/
├── sort-llm/                        # Main git repo (has remote)
│   ├── sortgpt_toolkit/             # Core library (model, training, intervention)
│   │   ├── model.py                 # GPT model class, data generation, checkpoint loading
│   │   ├── intervene.py             # Attention storage, intervention, ablation analysis
│   │   ├── train.py                 # Training script
│   │   └── ...
│   └── mechanistic-interpretability/ # Committed analysis results (copy of top-level work)
├── plot_*.py                        # 29 analysis/plotting scripts
├── run_grid.py                      # Training orchestrator (8 configs × 8 GPUs)
├── new-grid/                        # Training outputs
│   ├── k{16,32}_N{128,256,512,1024}/
│   │   ├── checkpoints/             # .pt files (NOT committed)
│   │   ├── plots/                   # Generated analysis plots
│   │   └── logs/                    # Training and plotting logs
├── REPORT.md                        # Human-readable findings
└── AGENT_CONTEXT.md                 # This file
```

## 2. Model Architecture

The model is defined in `sort-llm/sortgpt_toolkit/model.py`. Key classes:

### GPTConfig (dataclass)
```python
@dataclass
class GPTConfig:
    block_size: int       # Number of unsorted tokens (k). E.g., 32
    vocab_size: int       # vocab_n + 1 (includes SEP token). E.g., 513 for N=512
    n_layers: int         # Number of transformer blocks. Always 2 in our experiments
    n_heads: int          # Number of attention heads. Always 1
    n_embd: int           # Embedding dimension. Always 64
    without_pos: bool     # Whether to disable positional embeddings
    use_mlp: bool         # Whether blocks include MLP
    use_final_LN: bool    # Whether to apply final LayerNorm before lm_head
    max_seq_len: int      # Maximum sequence length for positional embeddings
```

### GPT Model
- **Embedding**: `wte` (token) + `wpe` (positional), weight-tied with `lm_head`
- **Blocks** (pre-norm): `x = x + attn(ln_1(x))`, then `x = x + mlp(ln_2(x))`
- **Attention**: Single head, `c_attn` projects to Q/K/V (3×n_embd), `c_proj` projects back
- **MLP**: `fc_1` (n_embd → 3×n_embd) → GELU → `fc_2` (3×n_embd → n_embd)
- **Forward signature**: `model(idx, block_size=block_size)` — block_size is REQUIRED as kwarg
- **Causal mask**: Applied inside attention (causal=True)

### Weight Access Patterns
```python
block0 = model.transformer.h[0]  # Layer 0
block1 = model.transformer.h[1]  # Layer 1

# Attention weights for a block:
# c_attn.weight shape: (3*n_embd, n_embd) — [W_q; W_k; W_v] stacked
# c_attn.bias shape: (3*n_embd,)
W_q = block.attn.c_attn.weight[:n_embd, :]
b_q = block.attn.c_attn.bias[:n_embd]
W_k = block.attn.c_attn.weight[n_embd:2*n_embd, :]
b_k = block.attn.c_attn.bias[n_embd:2*n_embd]
W_v = block.attn.c_attn.weight[2*n_embd:3*n_embd, :]
b_v = block.attn.c_attn.bias[2*n_embd:3*n_embd]

# Projection after attention:
# c_proj.weight shape: (n_embd, n_embd)
# c_proj.bias shape: (n_embd,)

# MLP weights:
# block.mlp.fc_1.weight: (3*n_embd, n_embd)
# block.mlp.fc_2.weight: (n_embd, 3*n_embd)

# Embeddings:
e_all = model.transformer.wte.weight[:vocab_n]  # (N, C), excludes SEP
```

## 3. Data Format

### Input Sequence
`[unsorted_tokens | SEP | sorted_tokens]` — total length `2*block_size + 1`

- Positions `0..block_size-1`: unsorted input tokens (random permutation of k values from 0..vocab_n-1)
- Position `block_size`: SEP token (value = vocab_n)
- Positions `block_size+1..2*block_size`: sorted output tokens (ascending order)

### Prediction Task
At position `block_size + p` (containing `sorted[p]`), the model predicts `sorted[p+1]`.
This means the model's job at each sorted output position is to predict the **next** sorted value.

### Batch Generation
```python
from model import DEVICE, load_model_from_checkpoint, get_batch

# Standard random batch:
idx = get_batch(batch_size=1, length=block_size, device=DEVICE, vocab_n=vocab_n)
# Returns shape (1, 2*block_size+1)

# vocab_n vs vocab_size:
vocab_n = model.config.vocab_size - 1  # Number range: 0..vocab_n-1
# vocab_size includes the SEP token
```

### Consecutive Batch (defined in plot scripts, not in toolkit)
```python
def generate_consecutive_batch(block_size, vocab_n, consec_len, device):
    """Generate a sample with `consec_len` consecutive numbers guaranteed."""
    # Picks random start i, ensures i..i+consec_len-1 all present
    # Fills remaining positions with random non-consecutive values
```

## 4. Loading and Running Models

### Checkpoint Loading
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

from model import DEVICE, load_model_from_checkpoint, get_batch
from intervene import enable_attention_storage

model = load_model_from_checkpoint("new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt")
block_size = model.config.block_size   # 32
vocab_n = model.config.vocab_size - 1  # 512
n_embd = model.config.n_embd           # 64
```

### Checkpoint Path Convention
`new-grid/k{K}_N{N}/checkpoints/std0p01_iseed1__ckpt{ITER}.pt`

Available iterations: 5000, 10000, 15000, ..., 95000, 100000

### Accessing Attention Weights
The toolkit's attention uses `F.scaled_dot_product_attention` (fused, no weight storage).
To access attention weights, you must patch the forward:

```python
from intervene import enable_attention_storage

enable_attention_storage(model)
model(idx, block_size=block_size)

# Post-softmax attention: model.transformer.h[layer].attn.attn  — shape (T, T)
# Pre-softmax attention: model.transformer.h[layer].attn.raw_attn — shape (T, T)
```

### CLI Pattern for Scripts
```bash
python plot_qk_deep_decomp.py \
    --ckpt new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt \
    --output new-grid/k32_N512/plots/qk_deep_decomp.png \
    --n-trials 400
```

## 5. Key Helper Functions (from plot_qk_deep_decomp.py)

### get_components: Extract Residual Stream Parts
```python
@torch.no_grad()
def get_components(model, idx):
    """Manual forward through Block 0, returning each residual component."""
    block0 = model.transformer.h[0]
    B, T = idx.size()
    pos = model.transformer.wpe(model.pos_idx[:T])
    embed = model.transformer.wte(idx) + pos          # token + positional embedding

    h0 = block0.ln_1(embed)
    attn0_out = block0.attn(h0)                        # L1 attention output (after c_proj)
    x_after_attn0 = embed + attn0_out

    h0_mlp = block0.ln_2(x_after_attn0)
    mlp0_out = block0.mlp(h0_mlp)                      # L1 MLP output

    return embed, attn0_out, mlp0_out
    # Full residual = embed + attn0_out + mlp0_out
```

### compute_l2_attention: Get L2 Attention from Modified Residual
```python
@torch.no_grad()
def compute_l2_attention(model, residual):
    """Compute L2 attention weights from a (possibly modified) residual."""
    block1 = model.transformer.h[1]
    h = block1.ln_1(residual)
    B, T, C = h.size()
    qkv = block1.attn.c_attn(h)
    q, k, _ = qkv.split(n_embd, dim=2)
    # ... applies causal mask and softmax ...
    return att  # (T, T)
```

### build_residual: Ablation Conditions
Modifies residual ONLY at unsorted positions (keys, 0..block_size-1):
- `"normal"`: embed + attn0_out + mlp0_out (unchanged)
- `"zero_mlp_keys"`: zero MLP at key positions
- `"zero_attn_causal"`: zero attn0 AND recompute MLP from embed only
- `"zero_attn_direct"`: zero attn0 in residual but keep original MLP
- `"mlp_embed_only"`: MLP receives LN(embed) instead of LN(embed+attn0)
- `"mlp_attn_only"`: MLP receives LN(attn0) instead of LN(embed+attn0)
- `"mlp_only"`: zero embed+attn0 in residual, keep only MLP
- `"mlp_only_from_embed"`: same but MLP recomputed from embed only

### L1 Value Vector Computation
```python
e_all = model.transformer.wte.weight[:vocab_n]  # (N, C)
ln1_e = block0.ln_1(e_all)
W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
v = ln1_e @ W_v.T + b_v
V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias  # (N, C)
# V_all[x] = L1 value vector when attention fully focused on token x
```

## 6. Measurement Methodology

### Correct Target Mapping (CRITICAL — previous bug was here)
```python
sorted_tokens = idx[0, block_size+1:].cpu().numpy()
unsorted_tokens = idx[0, :block_size].cpu().numpy()

val_to_pos = {}
for p in range(block_size):
    val_to_pos[int(unsorted_tokens[p])] = p

for p in range(block_size - 1):  # NOT block_size
    query_pos = block_size + 1 + p   # position containing sorted[p]
    target_val = int(sorted_tokens[p + 1])  # predicts sorted[p+1], NOT sorted[p]
    if target_val not in val_to_pos:
        continue
    key_pos = val_to_pos[target_val]
    attn_weight = l2_attn[query_pos, key_pos].item()
```

**Off-by-one pitfall**: The query at position `block_size + 1 + p` contains `sorted[p]`
and predicts `sorted[p+1]`. An earlier version incorrectly used `sorted[p]` as the
target, yielding ~7% accuracy instead of the correct ~80%.

### Scope of Ablation (CRITICAL — previous bug was here)
When testing "embed-only MLP" effects, modifications must be applied ONLY to the
positions being tested (keys or queries), not globally. Applying MLP modifications
to ALL positions simultaneously changes both keys AND queries, giving misleading results.

**Correct (key-side only)**:
```python
mlp_embed = block0.mlp(block0.ln_2(embed))
mod_mlp = mlp0_out.clone()
mod_mlp[:, :block_size, :] = mlp_embed[:, :block_size, :]  # ONLY unsorted positions
residual = embed + attn0_out + mod_mlp
```

**Incorrect (would also affect queries)**:
```python
mlp_embed = block0.mlp(block0.ln_2(embed))
residual = embed + attn0_out + mlp_embed  # Modifies ALL positions!
```

## 7. Primary Model Configuration (k=32, N=512)

- `block_size = 32` (sequence length of unsorted/sorted parts)
- `vocab_size = 513` (512 numbers + 1 SEP token)
- `n_layers = 2`, `n_heads = 1`, `n_embd = 64`
- Total sequence length: `2*32 + 1 = 65`
- Trained for 100k iterations with lr=0.03, init_std=0.01

### Key Numerical Results (100k checkpoint)
- Normal L2 attention on correct key: **0.805** (mean weight)
- Normal Top-1 accuracy: **99.5%**
- Zero MLP at keys: **0.002** attn, **7.2%** top-1
- MLP only (zero embed+attn in residual): **0.805** attn, **99.5%** top-1
- MLP from embed only: **0.230** attn, **70.4%** top-1
- L1 self-attention at unsorted positions: **24%** (cross-attention: **76%**)
- Cosine similarity (normal MLP vs embed-only MLP): median **0.956**

## 8. Grid of Models Trained

| Config | block_size (k) | vocab_n (N) | GPU |
|--------|---------------|-------------|-----|
| k16_N128 | 16 | 128 | 0 |
| k16_N256 | 16 | 256 | 1 |
| k16_N512 | 16 | 512 | 2 |
| k16_N1024 | 16 | 1024 | 3 |
| k32_N128 | 32 | 128 | 4 |
| k32_N256 | 32 | 256 | 5 |
| k32_N512 | 32 | 512 | 6 |
| k32_N1024 | 32 | 1024 | 7 |

All use: `init_std=0.01, lr=0.03, seed=1, n_embd=64, 2 layers, 1 head`
Checkpoints every 5k iterations up to 100k.

## 9. Known Pitfalls and Past Bugs

1. **Off-by-one in target mapping**: Position `block_size+1+p` predicts `sorted[p+1]`,
   not `sorted[p]`. Getting this wrong yields ~7% accuracy vs the correct ~80%.

2. **Ablation scope**: When testing MLP input modifications, apply ONLY to the positions
   being tested (keys OR queries), not both. Applying to all positions simultaneously
   conflates key-side and query-side effects, giving ~80% "attn helps" instead of ~30%.

3. **LayerNorm nonlinearity**: LN(a+b) ≠ LN(a) + LN(b). When decomposing QK scores
   by residual component, applying LN separately to each component gives scores that
   do NOT sum to the full score. Interpret relative magnitudes, not exact sums.

4. **enable_attention_storage**: Must be called before accessing `.attn` or `.raw_attn`
   on attention modules. Without it, these attributes don't exist (the default forward
   uses fused SDPA which doesn't store weights).

5. **model() requires block_size kwarg**: `model(idx, block_size=block_size)` — forgetting
   `block_size=` causes a TypeError. The forward also asserts `T == 2*block_size + 1`.

6. **vocab_n vs vocab_size**: `vocab_n = model.config.vocab_size - 1`. Token values are
   `0..vocab_n-1`. The SEP token has value `vocab_n`. When generating embeddings for
   all "real" tokens, use `wte.weight[:vocab_n]`, not `wte.weight[:vocab_size]`.

## 10. What Has NOT Been Investigated Yet

- **Output logits pathway**: How the L2 output + ln_f + lm_head converts attention
  patterns into correct token predictions. The OV circuit of L2 is unexplored.
- **Training dynamics**: How the sorting circuit forms during training (only 60k vs 100k
  checkpoints were compared; they show similar structure).
- **Generalization across configs**: Deep analysis was only done for k32_N512. Other
  configs (k16, N=128/256/1024) have basic plots but not the detailed ablation studies.
- **Edge cases**: Behavior on inputs with near-duplicates (values differing by 1) where
  the model is most fragile. The importance analysis showed these are the hardest cases
  but didn't dig into the specific mechanism used.
- **Positional encoding role**: How positional embeddings interact with the sorting
  circuit (they're nonzero in these models).
- **MLP internal structure**: What the MLP neurons actually compute — which neurons
  activate for which tokens, whether there are interpretable features.
