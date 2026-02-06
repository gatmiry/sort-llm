#!/usr/bin/env python3
"""
Plots to highlight:
1) In-distribution performance (train range) vs OOD generalization (K=17-32)
2) The importance / effect size of LayerNorm (LN on vs off)

Reads local W&B outputs under:
  <grid_root>/**/wandb-summary.json  with sibling config.yaml

Outputs (by default) into:
  length_generalization/results/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class Run:
    training: str  # dynamic | fixed
    n_layers: int
    use_mlp: bool
    length_mode: str  # mix | curriculum | fixed
    train_min_k: int
    train_max_k: int
    use_ln: bool
    train_em: Optional[float]
    in_dist_test_em: Optional[float]  # mean over test/Kk within train range
    gen_em: Optional[float]  # mean gen exact-match over K=17..32


def safe_value(cfg: dict, key: str, default=None):
    v = cfg.get(key, None)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return default


def mean_over_keys(summary: dict, keys: List[str]) -> Optional[float]:
    xs = [float(summary[k]) for k in keys if k in summary]
    if not xs:
        return None
    return float(np.mean(xs))


def load_runs(grid_root: str) -> List[Run]:
    runs: List[Run] = []
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

        # only compare on our canonical OOD test protocol
        test_min_k = int(safe_value(cfg, "test_min_k", 17) or 17)
        test_max_k = int(safe_value(cfg, "test_max_k", 32) or 32)
        if test_min_k != 17 or test_max_k != 32:
            continue

        n_layers = safe_value(cfg, "n_layers", None)
        use_mlp = safe_value(cfg, "use_mlp", None)
        length_mode = safe_value(cfg, "length_mode", None)
        train_min_k = safe_value(cfg, "train_min_k", None)
        train_max_k = safe_value(cfg, "train_max_k", None)
        use_ln = bool(safe_value(cfg, "use_layernorm", True))

        if None in (n_layers, use_mlp, length_mode, train_min_k, train_max_k):
            continue

        n_layers = int(n_layers)
        use_mlp = bool(use_mlp)
        length_mode = str(length_mode).strip().lower()
        train_min_k = int(train_min_k)
        train_max_k = int(train_max_k)

        if train_min_k == train_max_k:
            training = "fixed"
            lm = "fixed"
        else:
            training = "dynamic"
            if length_mode not in ("mix", "curriculum"):
                continue
            lm = length_mode

        train_em = summ.get("train/exact_match_acc", None)
        train_em = float(train_em) if train_em is not None else None

        # in-dist evaluation keys are stored as test/Kk/* (within train range)
        in_keys = [f"test/K{k}/exact_match_acc" for k in range(train_min_k, train_max_k + 1)]
        in_dist_test_em = mean_over_keys(summ, in_keys)

        gen_keys = [f"gen/K{k}/exact_match_acc" for k in range(17, 33)]
        gen_em = mean_over_keys(summ, gen_keys)

        runs.append(
            Run(
                training=training,
                n_layers=n_layers,
                use_mlp=use_mlp,
                length_mode=lm,
                train_min_k=train_min_k,
                train_max_k=train_max_k,
                use_ln=use_ln,
                train_em=train_em,
                in_dist_test_em=in_dist_test_em,
                gen_em=gen_em,
            )
        )
    return runs


def plot_scatter_train_vs_gen(runs: List[Run], out_path: str) -> None:
    # Use in-dist test EM on x-axis (more meaningful than train EM), gen EM on y-axis.
    xs_on, ys_on = [], []
    xs_off, ys_off = [], []
    for r in runs:
        if r.in_dist_test_em is None or r.gen_em is None:
            continue
        if r.use_ln:
            xs_on.append(r.in_dist_test_em)
            ys_on.append(r.gen_em)
        else:
            xs_off.append(r.in_dist_test_em)
            ys_off.append(r.gen_em)

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.scatter(xs_off, ys_off, s=14, alpha=0.35, label="LN off", color="#d62728")
    plt.scatter(xs_on, ys_on, s=14, alpha=0.35, label="LN on", color="#1f77b4")
    plt.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("In-distribution exact-match (avg test/K within train range)")
    plt.ylabel("OOD gen exact-match (mean K17–32)")
    plt.title("In-distribution fit does not imply length generalization")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _group_mean(runs: List[Run], *, training: str) -> Dict[Tuple, Tuple[float, float, int]]:
    # group by (L, MLP, mode, train_max, LN)
    d: Dict[Tuple, List[float]] = {}
    for r in runs:
        if r.training != training:
            continue
        if r.gen_em is None:
            continue
        key = (r.n_layers, r.use_mlp, r.length_mode, r.train_max_k, r.use_ln)
        d.setdefault(key, []).append(r.gen_em)
    out = {}
    for k, vals in d.items():
        out[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


def plot_ln_effect_size_dynamic(runs: List[Run], out_path: str) -> None:
    # Effect size: (LN on mean gen) - (LN off mean gen)
    ag = _group_mean(runs, training="dynamic")
    # collect by setting without LN
    settings = sorted({(L, mlp, mode, kmax) for (L, mlp, mode, kmax, ln) in ag.keys()})

    # plot separate panels for L/MLP, each with mix+curr curves of delta vs Kmax
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm effect size on OOD generalization (dynamic training)", fontweight="bold")

    panels = [
        (1, False, "L1, MLP off"),
        (1, True, "L1, MLP on"),
        (2, False, "L2, MLP off"),
        (2, True, "L2, MLP on"),
    ]

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for mode, color, marker in [("mix", "#1f77b4", "o"), ("curriculum", "#ff7f0e", "s")]:
            xs, ys = [], []
            for (_L, _mlp, _mode, kmax) in settings:
                if (_L, _mlp, _mode) != (L, mlp, mode):
                    continue
                on = ag.get((L, mlp, mode, kmax, True), None)
                off = ag.get((L, mlp, mode, kmax, False), None)
                if on is None or off is None:
                    continue
                xs.append(kmax)
                ys.append(on[0] - off[0])
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, label=f"{mode} (LN_on - LN_off)")

        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("train_max_k")
        ax.set_ylabel("Δ gen exact-match (mean K17–32)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default="/temp_work/ch225816/sort-llm/grid_outputs")
    ap.add_argument(
        "--output-dir",
        default="/temp_work/ch225816/sort-llm-repo/length_generalization/results",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.grid_root)

    scatter_path = os.path.join(args.output_dir, "len_gen_in_dist_vs_ood_scatter.png")
    effect_path = os.path.join(args.output_dir, "len_gen_layernorm_effect_size_dynamic.png")

    plot_scatter_train_vs_gen(runs, scatter_path)
    plot_ln_effect_size_dynamic(runs, effect_path)

    print(f"Wrote: {scatter_path}")
    print(f"Wrote: {effect_path}")


if __name__ == "__main__":
    main()

