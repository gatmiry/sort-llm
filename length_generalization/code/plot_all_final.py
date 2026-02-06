#!/usr/bin/env python3
"""
Final comprehensive plotting for length generalization experiments.

Reads all W&B local outputs and produces a clean set of figures covering:
  1. Main result: Gen EM vs train_max_k (2x2: layers x MLP, curves for mix/curriculum/fixed, LN on only)
  2. LayerNorm comparison: LN on vs off (dynamic + fixed)
  3. In-dist vs OOD scatter (LN on vs off)
  4. Summary CSV and min-K threshold table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.rcParams.update({"font.size": 11})


# ─── data loading ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    n_layers: int
    use_mlp: bool
    use_ln: bool
    length_mode: str   # mix | curriculum | fixed
    training: str      # dynamic | fixed
    train_max_k: int
    train_em: Optional[float]
    in_dist_em: Optional[float]
    gen_em: Optional[float]


def _val(cfg, key, default=None):
    v = cfg.get(key)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return default


def load_runs(grid_root: str) -> List[Run]:
    runs = []
    for sp in glob.glob(os.path.join(grid_root, "**/wandb-summary.json"), recursive=True):
        cp = os.path.join(os.path.dirname(sp), "config.yaml")
        if not os.path.isfile(cp):
            continue
        try:
            cfg = yaml.safe_load(open(cp))
            summ = json.load(open(sp))
        except Exception:
            continue

        tmin_k = int(_val(cfg, "test_min_k", 17) or 17)
        tmax_k = int(_val(cfg, "test_max_k", 32) or 32)
        if tmin_k != 17 or tmax_k != 32:
            continue

        L   = _val(cfg, "n_layers")
        mlp = _val(cfg, "use_mlp")
        ln  = _val(cfg, "use_layernorm", True)
        mode = _val(cfg, "length_mode")
        kmin = _val(cfg, "train_min_k")
        kmax = _val(cfg, "train_max_k")
        if None in (L, mlp, mode, kmin, kmax):
            continue

        L, mlp, ln = int(L), bool(mlp), bool(ln)
        kmin, kmax = int(kmin), int(kmax)
        mode = str(mode).strip().lower()

        if kmin == kmax:
            training, lm = "fixed", "fixed"
        else:
            training = "dynamic"
            lm = mode if mode in ("mix", "curriculum") else None
            if lm is None:
                continue

        train_em = summ.get("train/exact_match_acc")
        train_em = float(train_em) if train_em is not None else None

        ids = [f"test/K{k}/exact_match_acc" for k in range(kmin, kmax + 1)]
        id_vals = [float(summ[k]) for k in ids if k in summ]
        in_dist_em = float(np.mean(id_vals)) if id_vals else None

        gkeys = [f"gen/K{k}/exact_match_acc" for k in range(17, 33)]
        gvals = [float(summ[k]) for k in gkeys if k in summ]
        gen_em = float(np.mean(gvals)) if gvals else None

        runs.append(Run(L, mlp, ln, lm, training, kmax, train_em, in_dist_em, gen_em))
    return runs


# ─── aggregation helpers ────────────────────────────────────────────

AggKey = Tuple  # flexible

def agg_gen(runs: List[Run], key_fn) -> Dict[AggKey, Tuple[float, float, int]]:
    d: Dict[AggKey, List[float]] = defaultdict(list)
    for r in runs:
        if r.gen_em is None:
            continue
        d[key_fn(r)].append(r.gen_em)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()}


# ─── Figure 1: Main result (LN on only) ────────────────────────────

def fig_main(runs: List[Run], path: str):
    ag = agg_gen([r for r in runs if r.use_ln],
                 lambda r: (r.n_layers, r.use_mlp, r.length_mode, r.training, r.train_max_k))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("Length Generalization (with LayerNorm): Mean Gen EM (K17–32) vs Train Max K", fontweight="bold")

    panels = [(1, False, "L1, MLP off"), (1, True, "L1, MLP on"),
              (2, False, "L2, MLP off"), (2, True, "L2, MLP on")]

    series = [
        ("mix",        "dynamic", "#1f77b4", "o", "-",  "Dynamic Mix (K2→Kmax)"),
        ("curriculum", "dynamic", "#ff7f0e", "s", "-",  "Dynamic Curriculum"),
        ("fixed",      "fixed",   "#2ca02c", "^", "-",  "Fixed (single K)"),
    ]

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for mode, trn, col, mk, ls, label in series:
            xs, ys, ye = [], [], []
            for kmax in sorted({k[-1] for k in ag}):
                key = (L, mlp, mode, trn, kmax)
                if key not in ag:
                    continue
                m, s, _ = ag[key]
                xs.append(kmax); ys.append(m); ye.append(s)
            if xs:
                ax.errorbar(xs, ys, yerr=ye, color=col, marker=mk, linestyle=ls,
                            capsize=3, markersize=6, linewidth=1.5, label=label)
        ax.set_title(title); ax.set_xlabel("Train max K"); ax.grid(True, alpha=0.25)
        ax.axhline(0.95, color="gray", ls="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("Gen exact-match (K17–32)")
    axes[1][0].set_ylabel("Gen exact-match (K17–32)")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Figure 2: LayerNorm ON vs OFF (dynamic) ───────────────────────

def fig_ln_dynamic(runs: List[Run], path: str):
    ag = agg_gen([r for r in runs if r.training == "dynamic"],
                 lambda r: (r.n_layers, r.use_mlp, r.length_mode, r.use_ln, r.train_max_k))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm ON vs OFF — Dynamic Training, Gen EM (K17–32)", fontweight="bold")

    panels = [(1, False, "L1, MLP off"), (1, True, "L1, MLP on"),
              (2, False, "L2, MLP off"), (2, True, "L2, MLP on")]

    styles = {
        ("mix", True):        dict(color="#1f77b4", marker="o", ls="-",  label="Mix, LN on"),
        ("mix", False):       dict(color="#1f77b4", marker="o", ls="--", label="Mix, LN off"),
        ("curriculum", True): dict(color="#ff7f0e", marker="s", ls="-",  label="Curriculum, LN on"),
        ("curriculum", False):dict(color="#ff7f0e", marker="s", ls="--", label="Curriculum, LN off"),
    }

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for (mode, ln), sty in styles.items():
            xs, ys, ye = [], [], []
            for kmax in sorted({k[-1] for k in ag}):
                key = (L, mlp, mode, ln, kmax)
                if key not in ag:
                    continue
                m, s, _ = ag[key]
                xs.append(kmax); ys.append(m); ye.append(s)
            if xs:
                ax.errorbar(xs, ys, yerr=ye, capsize=3, markersize=6, linewidth=1.5, **sty)
        ax.set_title(title); ax.set_xlabel("Train max K"); ax.grid(True, alpha=0.25)
        ax.axhline(0.95, color="gray", ls="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("Gen exact-match (K17–32)")
    axes[1][0].set_ylabel("Gen exact-match (K17–32)")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Figure 3: LayerNorm ON vs OFF (fixed single-K) ────────────────

def fig_ln_fixed(runs: List[Run], path: str):
    ag = agg_gen([r for r in runs if r.training == "fixed"],
                 lambda r: (r.n_layers, r.use_mlp, r.use_ln, r.train_max_k))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm ON vs OFF — Fixed Single-K Training, Gen EM (K17–32)", fontweight="bold")

    panels = [(1, False, "L1, MLP off"), (1, True, "L1, MLP on"),
              (2, False, "L2, MLP off"), (2, True, "L2, MLP on")]

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for ln, col, ls, label in [(True, "#2ca02c", "-", "LN on"), (False, "#d62728", "--", "LN off")]:
            xs, ys, ye = [], [], []
            for kmax in sorted({k[-1] for k in ag}):
                key = (L, mlp, ln, kmax)
                if key not in ag:
                    continue
                m, s, _ = ag[key]
                xs.append(kmax); ys.append(m); ye.append(s)
            if xs:
                ax.errorbar(xs, ys, yerr=ye, color=col, marker="^", ls=ls,
                            capsize=3, markersize=6, linewidth=1.5, label=label)
        ax.set_title(title); ax.set_xlabel("Train K (single length)"); ax.grid(True, alpha=0.25)
        ax.axhline(0.95, color="gray", ls="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9, loc="lower right")
    axes[0][0].set_ylabel("Gen exact-match (K17–32)")
    axes[1][0].set_ylabel("Gen exact-match (K17–32)")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Figure 4: In-dist vs OOD scatter ──────────────────────────────

def fig_scatter(runs: List[Run], path: str):
    fig = plt.figure(figsize=(7, 6))
    for ln, col, label in [(True, "#1f77b4", "LN on"), (False, "#d62728", "LN off")]:
        xs = [r.in_dist_em for r in runs if r.use_ln == ln and r.in_dist_em is not None and r.gen_em is not None]
        ys = [r.gen_em     for r in runs if r.use_ln == ln and r.in_dist_em is not None and r.gen_em is not None]
        plt.scatter(xs, ys, s=16, alpha=0.4, color=col, label=label)
    plt.axhline(0.95, color="gray", ls="--", alpha=0.5)
    plt.xlabel("In-distribution exact-match (avg test/K within train range)")
    plt.ylabel("OOD gen exact-match (mean K17–32)")
    plt.title("In-distribution fit does NOT imply length generalization\n(each dot = one run)")
    plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25); plt.legend()
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Figure 5: LN effect size (dynamic) ────────────────────────────

def fig_ln_effect(runs: List[Run], path: str):
    ag = agg_gen([r for r in runs if r.training == "dynamic"],
                 lambda r: (r.n_layers, r.use_mlp, r.length_mode, r.use_ln, r.train_max_k))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("LayerNorm Effect Size: Δ Gen EM = (LN on) − (LN off), Dynamic Training", fontweight="bold")

    panels = [(1, False, "L1, MLP off"), (1, True, "L1, MLP on"),
              (2, False, "L2, MLP off"), (2, True, "L2, MLP on")]

    for i, (L, mlp, title) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        for mode, col, mk in [("mix", "#1f77b4", "o"), ("curriculum", "#ff7f0e", "s")]:
            xs, ys = [], []
            for kmax in sorted({k[-1] for k in ag}):
                on  = ag.get((L, mlp, mode, True,  kmax))
                off = ag.get((L, mlp, mode, False, kmax))
                if on and off:
                    xs.append(kmax); ys.append(on[0] - off[0])
            if xs:
                ax.plot(xs, ys, color=col, marker=mk, markersize=6, linewidth=1.5,
                        label=f"{mode} (LN on − LN off)")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_title(title); ax.set_xlabel("Train max K"); ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.1, 1.1); ax.legend(fontsize=9, loc="upper right")
    axes[0][0].set_ylabel("Δ gen exact-match")
    axes[1][0].set_ylabel("Δ gen exact-match")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── CSV: full summary ─────────────────────────────────────────────

def write_csv(runs: List[Run], path: str):
    ag = agg_gen(runs,
                 lambda r: (r.n_layers, r.use_mlp, r.use_ln, r.length_mode, r.training, r.train_max_k))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_layers","use_mlp","use_layernorm","length_mode","training","train_max_k",
                     "mean_gen_em","std_gen_em","n_runs"])
        for key in sorted(ag):
            L, mlp, ln, mode, trn, kmax = key
            m, s, n = ag[key]
            w.writerow([L, int(mlp), int(ln), mode, trn, kmax, f"{m:.4f}", f"{s:.4f}", n])


# ─── CSV: min-K thresholds ─────────────────────────────────────────

def write_thresholds(runs: List[Run], path: str, threshold: float = 0.95):
    ag = agg_gen(runs,
                 lambda r: (r.n_layers, r.use_mlp, r.use_ln, r.length_mode, r.training, r.train_max_k))
    groups: Dict[Tuple, List[Tuple[int, float]]] = defaultdict(list)
    for (L, mlp, ln, mode, trn, kmax), (m, _s, _n) in ag.items():
        groups[(L, mlp, ln, mode, trn)].append((kmax, m))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_layers","use_mlp","use_layernorm","length_mode","training",f"min_k_at_{threshold}"])
        for key in sorted(groups):
            best = None
            for kmax, m in sorted(groups[key]):
                if m >= threshold:
                    best = kmax
                    break
            w.writerow([*key, best if best else ""])


# ─── main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default="/temp_work/ch225816/sort-llm/grid_outputs")
    ap.add_argument("--output-dir", default="/temp_work/ch225816/sort-llm-repo/length_generalization/results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.grid_root)
    print(f"Loaded {len(runs)} runs")

    out = args.output_dir
    fig_main(runs,       os.path.join(out, "fig1_main_result.png"))
    fig_ln_dynamic(runs, os.path.join(out, "fig2_layernorm_dynamic.png"))
    fig_ln_fixed(runs,   os.path.join(out, "fig3_layernorm_fixed.png"))
    fig_scatter(runs,    os.path.join(out, "fig4_in_dist_vs_ood.png"))
    fig_ln_effect(runs,  os.path.join(out, "fig5_layernorm_effect_size.png"))
    write_csv(runs,      os.path.join(out, "summary.csv"))
    write_thresholds(runs, os.path.join(out, "min_k_thresholds.csv"))

    for f in sorted(os.listdir(out)):
        print(f"  {f}")
    print("Done.")


if __name__ == "__main__":
    main()
