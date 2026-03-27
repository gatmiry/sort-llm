"""
Show top-5 attention scores from layer 0 for the LN0 model,
V512_B32_LR1e-02_MI20000 checkpoint at itr20000.
"""
import sys, os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_analysis import GPT, GPTConfig


def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    config = GPTConfig(
        block_size=cfg['block_size'],
        vocab_size=cfg['vocab_size'],
        with_layer_norm=cfg['with_layer_norm'],
    )
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, config


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def region_label(pos, block_size):
    if pos < block_size:
        return "unsorted"
    elif pos == block_size:
        return "SEP"
    else:
        return "sorted"


def show_top5_for_sample(model, config, device, sample_idx, layer=0):
    block_size = config.block_size
    vocab_size = config.vocab_size
    idx = get_batch(vocab_size, block_size).to(device)

    with torch.no_grad():
        logits, loss = model(idx)

    attn = model.transformer.h[layer].c_attn.attn
    seq = idx[0].cpu().tolist()
    T = len(seq)

    unsorted_part = seq[:block_size]
    sep = seq[block_size]
    sorted_part = seq[block_size + 1:]

    print(f"\n{'='*90}")
    print(f"Sample {sample_idx}")
    print(f"{'='*90}")
    print(f"Unsorted ({len(unsorted_part)} tokens): {unsorted_part}")
    print(f"SEP: {sep}")
    print(f"Sorted   ({len(sorted_part)} tokens): {sorted_part}")

    predictions = torch.argmax(logits[0, block_size:2*block_size, :], dim=1).cpu().tolist()
    targets = seq[block_size+1:]
    correct = [p == t for p, t in zip(predictions, targets)]
    print(f"Accuracy: {sum(correct)}/{len(correct)}")

    for j in range(block_size, 2 * block_size):
        pos_in_sorted = j - block_size
        current_token = seq[j]
        target_token = seq[j + 1] if j + 1 < T else None

        attn_row = attn[j, :j+1].cpu().numpy()
        top5_indices = np.argsort(attn_row)[-5:][::-1]
        top5_scores = attn_row[top5_indices]

        pred = predictions[pos_in_sorted]
        is_correct = "CORRECT" if pred == target_token else "WRONG"

        print(f"\n  Position {j} (sorted[{pos_in_sorted}]): "
              f"current={current_token}, predicting next={target_token}, "
              f"pred={pred} [{is_correct}]")
        print(f"  Top-5 attention scores (layer {layer}):")
        for rank, (ki, sc) in enumerate(zip(top5_indices, top5_scores)):
            token_at_k = seq[ki]
            region = region_label(ki, block_size)
            marker = ""
            if token_at_k == target_token:
                marker = " <-- TARGET"
            print(f"    #{rank+1}: pos={ki:2d} ({region:>8s}) "
                  f"token={token_at_k:3d}  attn={sc:.4f}{marker}")


def main():
    device = 'cpu'
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

    ln_val = int(os.environ.get("LN_VAL", "0"))
    ln_label = "LN1 (with LayerNorm)" if ln_val == 1 else "LN0 (without LayerNorm)"
    ckpt_path = os.path.join(
        output_base,
        f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2",
        f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2_itr20000.pt"
    )
    print(f"Model: {ln_label}")
    print(f"Config: V=512, B=32, LR=1e-02, ckpt=20000")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")

    model, config = load_model(ckpt_path, device)
    print(f"block_size={config.block_size}, vocab_size={config.vocab_size}")

    torch.manual_seed(42)
    for i in range(3):
        show_top5_for_sample(model, config, device, sample_idx=i+1, layer=0)


if __name__ == '__main__':
    main()
