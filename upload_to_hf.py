#!/usr/bin/env python3
"""Upload new-grid-multiple-2 and new-grid-multiple-3 checkpoints to HuggingFace.

Maps local paths to the existing repo structure:
  local:  new-grid-multiple-{2,3}/k{K}_N{N}/seed{S}/checkpoints/*.pt
  remote: checkpoints/k{K}_N{N}/seed{S}/*.pt
"""
import os
import time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = "gatmiry/sortgpt-checkpoints"
BASE = Path("/mnt/task_runtime/sort-llm")
BATCH_DIRS = [
    BASE / "new-grid-multiple-2",
    BASE / "new-grid-multiple-3",
]

api = HfApi()

operations = []
for batch_dir in BATCH_DIRS:
    for pt_file in sorted(batch_dir.rglob("*.pt")):
        rel = pt_file.relative_to(batch_dir)
        parts = rel.parts  # e.g. ('k32_N512', 'seed6', 'checkpoints', 'std0p01_iseed6__ckpt5000.pt')
        config_dir = parts[0]   # k32_N512
        seed_dir = parts[1]     # seed6
        filename = parts[-1]    # std0p01_iseed6__ckpt5000.pt
        remote_path = f"checkpoints/{config_dir}/{seed_dir}/{filename}"

        operations.append(CommitOperationAdd(
            path_in_repo=remote_path,
            path_or_fileobj=str(pt_file),
        ))

print(f"Total files to upload: {len(operations)}")
print(f"Uploading to {REPO_ID} ...")

CHUNK_SIZE = 200
for i in range(0, len(operations), CHUNK_SIZE):
    chunk = operations[i:i+CHUNK_SIZE]
    lo = i + 1
    hi = min(i + CHUNK_SIZE, len(operations))
    label = f"Add checkpoints (seeds 6-25) [{lo}-{hi}/{len(operations)}]"
    print(f"\n{time.strftime('%H:%M:%S')} | Committing {label}")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=chunk,
        commit_message=label,
    )
    print(f"{time.strftime('%H:%M:%S')} | Done: {hi}/{len(operations)}")

print(f"\n{time.strftime('%H:%M:%S')} | ===== ALL UPLOADS COMPLETE =====")
