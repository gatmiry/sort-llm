#!/usr/bin/env python3
"""
Polished plotting for Huang-style length-range experiments.

Produces:
  - fig_huang_ranges_by_setting.png
  - fig_huang_variant_averages.png
  - huang_range_summary.csv

Runs are grouped by:
  variant x n_layers x use_mlp
and evaluated on:
  - id:      K=2..50
  - ood_mid: K=51..100
  - ood_long:K=101..150
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


@dataclass(frozen=True)
class Run:
    variant: str
    n_layers: int
    use_mlp: bool
    seed: int
    iter_done: int
    id_em: Optional[float]
    mid_em: Optional[float]
    long_em: Optional[float]
    train_em: Optional[float]


def _cfg_val(cfg: dict, key: str, default=None):
    v = cfg.get(key, default)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _mean_metric(summary: dict, prefix: str, k_min: int, k_max: int) -> Optional[float]:
    vals = []
    for k in range(int(k_min), int(k_max) + 1):
        kk = f"{prefix}/K{k}/exact_match_acc"
        if kk in summary:
            vals.append(float(summary[kk]))
    if not vals:
        return None
    return float(np.mean(vals))


def _detect_variant(cfg: dict) -> Optional[str]:
    use_ln = bool(_cfg_val(cfg, "use_layernorm", True))
    norm_type = str(_cfg_val(cfg, "norm_type", "layernorm")).strip().lower()
    train_min_k = int(_cfg_val(cfg, "train_min_k", -1))
    train_max_k = int(_cfg_val(cfg, "train_max_k", -1))
    qk = bool(_cfg_val(cfg, "use_qk_norm", False))
    qk_rms = bool(_cfg_val(cfg, "use_rms_qk_norm", False))
    gate_head = bool(_cfg_val(cfg, "use_gated_attention", False))
    gate_elem = bool(_cfg_val(cfg, "use_elementwise_attn_output_gate", False))

    if use_ln and train_min_k == 25 and train_max_k == 25 and norm_type == "layernorm":
        return "LN_fixed25"
    if use_ln and train_min_k == 50 and train_max_k == 50 and norm_type == "layernorm":
        return "LN_fixed50"
    if use_ln and train_min_k == 50 and train_max_k == 50 and norm_type == "rmsnorm":
        return "RMS_fixed50"
    if (not use_ln) and qk and qk_rms:
        return "noLN_rmsQkNorm"
    if (not use_ln) and gate_elem:
        return "noLN_elemGate"
    if (not use_ln) and gate_head:
        return "noLN_headGate"
    return None


def load_runs(grid_root: str) -> List[Run]:
    runs: List[Run] = []
    summary_paths = glob.glob(os.path.join(grid_root, "huang_*", "**", "wandb-summary.json"), recursive=True)
    for sp in summary_paths:
        cp = os.path.join(os.path.dirname(sp), "config.yaml")
        if not os.path.isfile(cp):
            continue
        try:
            with open(cp, "r") as f:
                cfg = yaml.safe_load(f)
            with open(sp, "r") as f:
                summary = json.load(f)
        except Exception:
            continue

        variant = _detect_variant(cfg)
        if variant is None:
            continue

        # Keep Huang-protocol-only runs.
        e_id_min = int(_cfg_val(cfg, "eval_id_min", -1))
        e_id_max = int(_cfg_val(cfg, "eval_id_max", -1))
        e_mid_min = int(_cfg_val(cfg, "eval_mid_min", -1))
        e_mid_max = int(_cfg_val(cfg, "eval_mid_max", -1))
        e_long_min = int(_cfg_val(cfg, "eval_long_min", -1))
        e_long_max = int(_cfg_val(cfg, "eval_long_max", -1))
        if (e_id_min, e_id_max, e_mid_min, e_mid_max, e_long_min, e_long_max) != (2, 50, 51, 100, 101, 150):
            continue

        n_layers = int(_cfg_val(cfg, "n_layers", -1))
        use_mlp = bool(_cfg_val(cfg, "use_mlp", False))
        seed = int(_cfg_val(cfg, "seed", -1))
        iter_done = int(summary.get("iter", -1))

        id_em = _mean_metric(summary, "test", 2, 50)
        mid_em = _mean_metric(summary, "gen_mid", 51, 100)
        long_em = _mean_metric(summary, "gen_long", 101, 150)
        train_em = summary.get("train/exact_match_acc")
        train_em = float(train_em) if train_em is not None else None

        # Only keep runs that reached at least one evaluation.
        if id_em is None and mid_em is None and long_em is None:
            continue

        runs.append(
            Run(
                variant=variant,
                n_layers=n_layers,
                use_mlp=use_mlp,
                seed=seed,
                iter_done=iter_done,
                id_em=id_em,
                mid_em=mid_em,
                long_em=long_em,
                train_em=train_em,
            )
        )
    return runs


def dedupe_latest_by_seed(runs: List[Run]) -> List[Run]:
    # Deduplicate retries/resubmissions: keep the run with the largest iter_done.
    best: Dict[Tuple[str, int, bool, int], Run] = {}
    for r in runs:
        k = (r.variant, r.n_layers, r.use_mlp, r.seed)
        old = best.get(k)
        if old is None or r.iter_done > old.iter_done:
            best[k] = r
    return list(best.values())


def summarize(runs: List[Run]):
    bucket: Dict[Tuple[str, int, bool], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        key = (r.variant, r.n_layers, r.use_mlp)
        if r.id_em is not None:
            bucket[key]["id"].append(r.id_em)
        if r.mid_em is not None:
            bucket[key]["mid"].append(r.mid_em)
        if r.long_em is not None:
            bucket[key]["long"].append(r.long_em)
        if r.train_em is not None:
            bucket[key]["train"].append(r.train_em)
    out = {}
    for key, v in bucket.items():
        out[key] = {}
        for metric in ("id", "mid", "long", "train"):
            arr = v.get(metric, [])
            if arr:
                out[key][metric] = (float(np.mean(arr)), float(np.std(arr)), len(arr))
            else:
                out[key][metric] = (None, None, 0)
    return out


def write_csv(summary: dict, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "n_layers",
                "use_mlp",
                "id_2_50_mean",
                "id_2_50_std",
                "mid_51_100_mean",
                "mid_51_100_std",
                "long_101_150_mean",
                "long_101_150_std",
                "train_em_mean",
                "train_em_std",
                "n_seeds",
            ]
        )
        for (variant, n_layers, use_mlp), vals in sorted(summary.items()):
            id_m, id_s, n1 = vals["id"]
            md_m, md_s, n2 = vals["mid"]
            lg_m, lg_s, n3 = vals["long"]
            tr_m, tr_s, n4 = vals["train"]
            n = max(n1, n2, n3, n4)
            w.writerow(
                [
                    variant,
                    n_layers,
                    int(use_mlp),
                    "" if id_m is None else f"{id_m:.4f}",
                    "" if id_s is None else f"{id_s:.4f}",
                    "" if md_m is None else f"{md_m:.4f}",
                    "" if md_s is None else f"{md_s:.4f}",
                    "" if lg_m is None else f"{lg_m:.4f}",
                    "" if lg_s is None else f"{lg_s:.4f}",
                    "" if tr_m is None else f"{tr_m:.4f}",
                    "" if tr_s is None else f"{tr_s:.4f}",
                    n,
                ]
            )


def _variant_color(v: str) -> str:
    return {
        "noLN_rmsQkNorm": "#1f77b4",
        "noLN_elemGate": "#ff7f0e",
        "LN_fixed25": "#8c564b",
        "LN_fixed50": "#2ca02c",
        "RMS_fixed50": "#17becf",
        "noLN_headGate": "#9467bd",
    }.get(v, "#7f7f7f")


def fig_by_setting(summary: dict, out_png: str):
    # 2x2 panels for (L1/L2) x (MLP off/on), bars are variants with 3 range groups.
    settings = [(1, False), (1, True), (2, False), (2, True)]
    ranges = [("id", "2-50"), ("mid", "51-100"), ("long", "101-150")]
    variants = ["noLN_rmsQkNorm", "noLN_elemGate", "LN_fixed25", "LN_fixed50", "RMS_fixed50", "noLN_headGate"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    fig.suptitle("Huang-Length Evaluation by Setting", fontweight="bold")

    x = np.arange(len(ranges))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(variants))

    for i, (L, mlp) in enumerate(settings):
        ax = axes[i // 2][i % 2]
        for vi, v in enumerate(variants):
            key = (v, L, mlp)
            vals = summary.get(key)
            if not vals:
                continue
            means = [vals[r][0] if vals[r][0] is not None else np.nan for r, _ in ranges]
            stds = [vals[r][1] if vals[r][1] is not None else 0.0 for r, _ in ranges]
            ax.bar(
                x + offsets[vi],
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=_variant_color(v),
                alpha=0.9,
                label=v,
            )
        ax.set_title(f"L{L}, MLP {'on' if mlp else 'off'}")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in ranges])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0.95, color="gray", ls="--", alpha=0.4)

    axes[0][0].set_ylabel("Exact-match accuracy")
    axes[1][0].set_ylabel("Exact-match accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_variant_average(summary: dict, out_png: str):
    # Average over layers+MLP to give a high-level variant comparison.
    ranges = ["id", "mid", "long"]
    range_labels = ["2-50", "51-100", "101-150"]
    variants = ["noLN_rmsQkNorm", "noLN_elemGate", "LN_fixed25", "LN_fixed50", "RMS_fixed50", "noLN_headGate"]

    agg = {v: {r: [] for r in ranges} for v in variants}
    for (v, _L, _mlp), vals in summary.items():
        if v not in agg:
            continue
        for r in ranges:
            if vals[r][0] is not None:
                agg[v][r].append(vals[r][0])

    fig = plt.figure(figsize=(11, 6))
    x = np.arange(len(range_labels))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(variants))

    for i, v in enumerate(variants):
        means = [float(np.mean(agg[v][r])) if agg[v][r] else np.nan for r in ranges]
        plt.bar(x + offsets[i], means, width=width, color=_variant_color(v), alpha=0.9, label=v)

    plt.xticks(x, range_labels)
    plt.ylim(-0.02, 1.02)
    plt.ylabel("Exact-match accuracy")
    plt.title("Variant-Level Averages Across All (Layer, MLP) Settings")
    plt.grid(axis="y", alpha=0.25)
    plt.axhline(0.95, color="gray", ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default="/temp_work/ch225816/sort-llm/grid_outputs")
    ap.add_argument("--output-dir", default="/temp_work/ch225816/sort-llm-repo/length_generalization/results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.grid_root)
    runs = dedupe_latest_by_seed(runs)
    summary = summarize(runs)

    out_csv = os.path.join(args.output_dir, "huang_range_summary.csv")
    out_fig1 = os.path.join(args.output_dir, "fig_huang_ranges_by_setting.png")
    out_fig2 = os.path.join(args.output_dir, "fig_huang_variant_averages.png")

    write_csv(summary, out_csv)
    fig_by_setting(summary, out_fig1)
    fig_variant_average(summary, out_fig2)

    print(f"Loaded runs (deduped): {len(runs)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_fig1}")
    print(f"Wrote: {out_fig2}")


if __name__ == "__main__":
    main()

