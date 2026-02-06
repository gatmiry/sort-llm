#!/usr/bin/env python3
"""
LayerNorm ON vs OFF comparison plots for length generalization experiments.

Reads W&B local outputs under grid_root:
  **/wandb-summary.json + sibling config.yaml

Generates:
  - len_gen_layernorm_dynamic.png : dynamic (train_min_k=2) curves
  - len_gen_layernorm_fixed.png   : fixed (train_min_k=train_max_k) curves
  - len_gen_layernorm_minK.csv    : min Kmax to reach >=0.95 mean gen EM
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class Key:
    training: str  # dynamic | fixed
    n_layers: int
    use_mlp: bool
    length_mode: str  # mix | curriculum | fixed
    train_max_k: int
    use_ln: bool


def safe_get(cfg: dict, name: str, default=None):
    v = cfg.get(name, None)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return default


def mean_gen_em(summary: dict, kmin: int = 17, kmax: int = 32) -> Optional[float]:
    accs = []
    for k in range(kmin, kmax + 1):
        kk = f"gen/K{k}/exact_match_acc"
        if kk in summary:
            accs.append(float(summary[kk]))
    if not accs:
        return None
    return float(np.mean(accs))


def load_points(grid_root: str) -> Dict[Key, List[float]]:
    points: Dict[Key, List[float]] = defaultdict(list)
    for summ_path in glob.glob(os.path.join(grid_root, "**/wandb-summary.json"), recursive=True):
        cfg_path = os.path.join(os.path.dirname(summ_path), "config.yaml")
        if not os.path.exists(cfg_path):
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            with open(summ_path, "r", encoding="utf-8") as f:
                summ = json.load(f)
        except Exception:
            continue

        test_min_k = int(safe_get(cfg, "test_min_k", 17) or 17)
        test_max_k = int(safe_get(cfg, "test_max_k", 32) or 32)
        # We only compare on the fixed OOD test protocol.
        if test_min_k != 17 or test_max_k != 32:
            continue

        n_layers = safe_get(cfg, "n_layers", None)
        use_mlp = safe_get(cfg, "use_mlp", None)
        length_mode = safe_get(cfg, "length_mode", None)
        train_min_k = safe_get(cfg, "train_min_k", None)
        train_max_k = safe_get(cfg, "train_max_k", None)
        use_ln = safe_get(cfg, "use_layernorm", True)  # default True for older runs

        if None in (n_layers, use_mlp, length_mode, train_min_k, train_max_k):
            continue

        n_layers = int(n_layers)
        use_mlp = bool(use_mlp)
        train_min_k = int(train_min_k)
        train_max_k = int(train_max_k)
        use_ln = bool(use_ln)
        length_mode = str(length_mode).strip().lower()

        # classify training regime
        if train_min_k == train_max_k:
            training = "fixed"
            lm = "fixed"
        else:
            training = "dynamic"
            if length_mode not in ("mix", "curriculum"):
                continue
            lm = length_mode

        m = mean_gen_em(summ, 17, 32)
        if m is None:
            continue

        points[Key(training, n_layers, use_mlp, lm, train_max_k, use_ln)].append(m)

    return points


def agg(points: Dict[Key, List[float]]) -> Dict[Key, Tuple[float, float, int]]:
    out: Dict[Key, Tuple[float, float, int]] = {}
    for k, vals in points.items():
        out[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


def plot_dynamic(ag: Dict[Key, Tuple[float, float, int]], out_path: str) -> None:
    ks = sorted({k.train_max_k for k in ag if k.training == "dynamic"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm ON vs OFF — Dynamic training (K=2→Kmax), test K=17–32", fontweight="bold")

    panels = [
        (1, False, "L1, MLP off"),
        (1, True, "L1, MLP on"),
        (2, False, "L2, MLP off"),
        (2, True, "L2, MLP on"),
    ]

    styles = {
        ("mix", True): dict(color="#1f77b4", marker="o", linestyle="-", label="mix, LN on"),
        ("mix", False): dict(color="#1f77b4", marker="o", linestyle="--", label="mix, LN off"),
        ("curriculum", True): dict(color="#ff7f0e", marker="s", linestyle="-", label="curriculum, LN on"),
        ("curriculum", False): dict(color="#ff7f0e", marker="s", linestyle="--", label="curriculum, LN off"),
    }

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for mode in ("mix", "curriculum"):
            for ln in (True, False):
                xs, ys, yerr = [], [], []
                for kmax in ks:
                    key = Key("dynamic", L, mlp, mode, kmax, ln)
                    if key not in ag:
                        continue
                    m, s, _n = ag[key]
                    xs.append(kmax)
                    ys.append(m)
                    yerr.append(s)
                if xs:
                    ax.errorbar(xs, ys, yerr=yerr, capsize=3, markersize=6, linewidth=1.5, **styles[(mode, ln)])

        ax.set_title(title)
        ax.set_xlabel("train_max_k")
        ax.grid(True, alpha=0.25)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="lower right")

    axes[0][0].set_ylabel("mean gen exact-match (K17–32)")
    axes[1][0].set_ylabel("mean gen exact-match (K17–32)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fixed(ag: Dict[Key, Tuple[float, float, int]], out_path: str) -> None:
    ks = sorted({k.train_max_k for k in ag if k.training == "fixed"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm ON vs OFF — Fixed single-K training, test K=17–32", fontweight="bold")

    panels = [
        (1, False, "L1, MLP off"),
        (1, True, "L1, MLP on"),
        (2, False, "L2, MLP off"),
        (2, True, "L2, MLP on"),
    ]

    styles = {
        True: dict(color="#2ca02c", marker="^", linestyle="-", label="LN on"),
        False: dict(color="#2ca02c", marker="^", linestyle="--", label="LN off"),
    }

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for ln in (True, False):
            xs, ys, yerr = [], [], []
            for kmax in ks:
                key = Key("fixed", L, mlp, "fixed", kmax, ln)
                if key not in ag:
                    continue
                m, s, _n = ag[key]
                xs.append(kmax)
                ys.append(m)
                yerr.append(s)
            if xs:
                ax.errorbar(xs, ys, yerr=yerr, capsize=3, markersize=6, linewidth=1.5, **styles[ln])

        ax.set_title(title)
        ax.set_xlabel("train_k (single length)")
        ax.grid(True, alpha=0.25)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="lower right")

    axes[0][0].set_ylabel("mean gen exact-match (K17–32)")
    axes[1][0].set_ylabel("mean gen exact-match (K17–32)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_minK_table(ag: Dict[Key, Tuple[float, float, int]], out_csv: str, threshold: float = 0.95) -> None:
    # For each setting without train_max_k: find smallest K reaching threshold
    # dynamic: (training,n_layers,use_mlp,length_mode,use_ln)
    # fixed:   (training,n_layers,use_mlp,use_ln)
    dyn_groups = defaultdict(list)
    fix_groups = defaultdict(list)
    for k, (m, _s, _n) in ag.items():
        if k.training == "dynamic":
            dyn_groups[(k.n_layers, k.use_mlp, k.length_mode, k.use_ln)].append((k.train_max_k, m))
        elif k.training == "fixed":
            fix_groups[(k.n_layers, k.use_mlp, k.use_ln)].append((k.train_max_k, m))

    lines = []
    lines.append("training,n_layers,use_mlp,length_mode,use_layernorm,min_train_max_k_at_0.95")

    for (L, mlp, mode, ln), vals in sorted(dyn_groups.items()):
        best = None
        for kmax, m in sorted(vals):
            if m >= threshold:
                best = kmax
                break
        lines.append(f"dynamic,{L},{int(mlp)},{mode},{int(ln)},{'' if best is None else best}")

    for (L, mlp, ln), vals in sorted(fix_groups.items()):
        best = None
        for kmax, m in sorted(vals):
            if m >= threshold:
                best = kmax
                break
        lines.append(f"fixed,{L},{int(mlp)},fixed,{int(ln)},{'' if best is None else best}")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default="/temp_work/ch225816/sort-llm/grid_outputs")
    ap.add_argument("--output-dir", default="/temp_work/ch225816/sort-llm-repo/length_generalization/results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pts = load_points(args.grid_root)
    ag = agg(pts)

    dyn_path = os.path.join(args.output_dir, "len_gen_layernorm_dynamic.png")
    fix_path = os.path.join(args.output_dir, "len_gen_layernorm_fixed.png")
    tab_path = os.path.join(args.output_dir, "len_gen_layernorm_minK.csv")

    plot_dynamic(ag, dyn_path)
    plot_fixed(ag, fix_path)
    write_minK_table(ag, tab_path, threshold=0.95)

    print(f"Wrote: {dyn_path}")
    print(f"Wrote: {fix_path}")
    print(f"Wrote: {tab_path}")


if __name__ == "__main__":
    main()

