import os
import re
import torch
from model_tbyt_3_withnormalization import GPT, GPTConfig

def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Loads a checkpoint from the Grid_training_without_duplicates folder.
    Parses the configuration from the filename.
    """
    filename = os.path.basename(checkpoint_path)
    
    # Pattern: Final_N128_K16_L1_H1_E16_r8over1_npos0_mlp0_dup0_testK16_iters53501.pt
    # Or: Final_vocab128_blockSize16_layers1_pos0_mlp0_embd16_dup0_mix0_iters53501.pt
    
    # Try pattern 1: Final_N{N}_K{K}_L{L}_H{H}_E{E}_r{r}_npos{npos}_mlp{mlp}_...
    match1 = re.search(r'Final_N(\d+)_K(\d+)_L(\d+)_H(\d+)_E(\d+)_.*_npos(\d+)_mlp(\d+)_', filename)
    # Try pattern 2: Final_vocab{N}_blockSize{K}_layers{L}_pos{npos}_mlp{mlp}_embd{E}_...
    match2 = re.search(r'Final_vocab(\d+)_blockSize(\d+)_layers(\d+)_pos(\d+)_mlp(\d+)_embd(\d+)_', filename)
    
    if match1:
        vocab_n = int(match1.group(1))
        block_size = int(match1.group(2))
        n_layers = int(match1.group(3))
        n_heads = int(match1.group(4))
        n_embd = int(match1.group(5))
        without_pos = bool(int(match1.group(6)))
        use_mlp = bool(int(match1.group(7)))
    elif match2:
        vocab_n = int(match2.group(1))
        block_size = int(match2.group(2))
        n_layers = int(match2.group(3))
        without_pos = bool(int(match2.group(4)))
        use_mlp = bool(int(match2.group(5)))
        n_embd = int(match2.group(6))
        n_heads = 1 # Default if not in filename pattern 2
    else:
        raise ValueError(f"Could not parse config from filename: {filename}")

    # total_vocab_size = vocab_n + 1 (number + SEP)
    vocab_size = vocab_n + 1
    
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_embd,
        without_pos=without_pos,
        use_mlp=use_mlp
    )
    
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if state_dict is nested under 'model' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Remove 'transformer.' prefix if present (sometimes happens with torch.save(model.state_dict()))
    # But usually the keys match exactly if we saved model.state_dict()
    
    # Handle possible torch._compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '') 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, config
