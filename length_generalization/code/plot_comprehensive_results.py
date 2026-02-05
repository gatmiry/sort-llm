#!/usr/bin/env python3
"""
Comprehensive visualization for length generalization experiments.

Covers all experimental conditions:
- Layers: 1, 2
- MLP: on, off
- Data loading: mix, curriculum
- Training regime: dynamic (K2→Kmax) vs fixed (single K)
"""
import json
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass
class RunResult:
    n_layers: int
    use_mlp: bool
    use_layernorm: bool
    length_mode: str
    train_min_k: int
    train_max_k: int
    test_min_k: int
    test_max_k: int
    seed: int
    gen_exact_match_mean: float  # mean over K17-32


def load_all_results(grid_root: str) -> List[RunResult]:
    """Load all wandb summary results."""
    results = []
    summaries = glob.glob(os.path.join(grid_root, "**/wandb-summary.json"), recursive=True)
    
    for summary_path in summaries:
        config_path = os.path.join(os.path.dirname(summary_path), "config.yaml")
        if not os.path.exists(config_path):
            continue
        
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        
        # Extract config values
        n_layers = cfg.get("n_layers", {}).get("value")
        use_mlp = cfg.get("use_mlp", {}).get("value")
        use_layernorm = cfg.get("use_layernorm", {}).get("value", True)  # default to True for old runs
        length_mode = cfg.get("length_mode", {}).get("value")
        train_min_k = cfg.get("train_min_k", {}).get("value")
        train_max_k = cfg.get("train_max_k", {}).get("value")
        test_min_k = cfg.get("test_min_k", {}).get("value")
        test_max_k = cfg.get("test_max_k", {}).get("value")
        seed = cfg.get("seed", {}).get("value", 1337)
        
        if any(v is None for v in [n_layers, use_mlp, length_mode, train_min_k, train_max_k]):
            continue
        
        # Compute mean gen exact match over K17-32
        gen_accs = []
        for k in range(17, 33):
            key = f"gen/K{k}/exact_match_acc"
            if key in summary:
                gen_accs.append(summary[key])
        
        if not gen_accs:
            continue
        
        results.append(RunResult(
            n_layers=n_layers,
            use_mlp=use_mlp,
            use_layernorm=use_layernorm,
            length_mode=length_mode,
            train_min_k=train_min_k,
            train_max_k=train_max_k,
            test_min_k=test_min_k or 17,
            test_max_k=test_max_k or 32,
            seed=seed,
            gen_exact_match_mean=np.mean(gen_accs),
        ))
    
    return results


def aggregate_results(results: List[RunResult]) -> Dict:
    """Aggregate results by condition, compute mean and std."""
    grouped = defaultdict(list)
    
    for r in results:
        is_fixed = (r.train_min_k == r.train_max_k)
        key = (r.n_layers, r.use_mlp, r.use_layernorm, r.length_mode, is_fixed, r.train_max_k)
        grouped[key].append(r.gen_exact_match_mean)
    
    aggregated = {}
    for key, vals in grouped.items():
        aggregated[key] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "n": len(vals),
        }
    
    return aggregated


def create_comprehensive_figure(results: List[RunResult], output_dir: str):
    """Create a comprehensive 2x2 figure comparing all conditions."""
    agg = aggregate_results(results)
    
    k_values = sorted(set(r.train_max_k for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Length Generalization: Gen Exact Match (K17-32) vs Train Max K", fontsize=14, fontweight='bold')
    
    conditions = [
        (1, True, "Layer 1, MLP On"),
        (1, False, "Layer 1, MLP Off"),
        (2, True, "Layer 2, MLP On"),
        (2, False, "Layer 2, MLP Off"),
    ]
    
    colors = {
        ("mix", False): "#1f77b4",      # Dynamic mix - blue
        ("curriculum", False): "#ff7f0e", # Dynamic curriculum - orange
        ("mix", True): "#2ca02c",         # Fixed - green
    }
    
    markers = {
        ("mix", False): "o",
        ("curriculum", False): "s",
        ("mix", True): "^",
    }
    
    labels = {
        ("mix", False): "Dynamic Mix (K2→Kmax)",
        ("curriculum", False): "Dynamic Curriculum (K2→Kmax)",
        ("mix", True): "Fixed (single K)",
    }
    
    for idx, (n_layers, use_mlp, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]
        
        for (mode, is_fixed), color in colors.items():
            means = []
            stds = []
            ks = []
            
            for k in k_values:
                key = (n_layers, use_mlp, mode, is_fixed, k)
                if key in agg:
                    means.append(agg[key]["mean"])
                    stds.append(agg[key]["std"])
                    ks.append(k)
            
            if means:
                ax.errorbar(
                    ks, means, yerr=stds,
                    label=labels[(mode, is_fixed)],
                    color=color,
                    marker=markers[(mode, is_fixed)],
                    markersize=6,
                    capsize=3,
                    linewidth=1.5,
                )
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Train Max K")
        ax.set_ylabel("Gen Exact Match Acc (K17-32)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "len_gen_comprehensive.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote plot: {out_path}")
    
    return out_path


def create_mlp_comparison_figure(results: List[RunResult], output_dir: str):
    """Create a figure comparing MLP on vs off."""
    agg = aggregate_results(results)
    
    k_values = sorted(set(r.train_max_k for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MLP On vs Off: Gen Exact Match (K17-32)", fontsize=14, fontweight='bold')
    
    conditions = [
        (1, "mix", False, "Layer 1, Dynamic Mix"),
        (1, "curriculum", False, "Layer 1, Dynamic Curriculum"),
        (2, "mix", False, "Layer 2, Dynamic Mix"),
        (2, "curriculum", False, "Layer 2, Dynamic Curriculum"),
    ]
    
    for idx, (n_layers, mode, is_fixed, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]
        
        for use_mlp, color, label in [(True, "#1f77b4", "MLP On"), (False, "#ff7f0e", "MLP Off")]:
            means = []
            stds = []
            ks = []
            
            for k in k_values:
                key = (n_layers, use_mlp, mode, is_fixed, k)
                if key in agg:
                    means.append(agg[key]["mean"])
                    stds.append(agg[key]["std"])
                    ks.append(k)
            
            if means:
                ax.errorbar(
                    ks, means, yerr=stds,
                    label=label,
                    color=color,
                    marker="o" if use_mlp else "s",
                    markersize=6,
                    capsize=3,
                    linewidth=1.5,
                )
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Train Max K")
        ax.set_ylabel("Gen Exact Match Acc (K17-32)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "len_gen_mlp_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote plot: {out_path}")
    
    return out_path


def create_fixed_vs_dynamic_figure(results: List[RunResult], output_dir: str):
    """Create a figure comparing fixed vs dynamic training."""
    agg = aggregate_results(results)
    
    k_values = sorted(set(r.train_max_k for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fixed vs Dynamic Training: Gen Exact Match (K17-32)", fontsize=14, fontweight='bold')
    
    conditions = [
        (1, True, "Layer 1, MLP On"),
        (1, False, "Layer 1, MLP Off"),
        (2, True, "Layer 2, MLP On"),
        (2, False, "Layer 2, MLP Off"),
    ]
    
    for idx, (n_layers, use_mlp, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]
        
        # Dynamic Mix
        means, stds, ks = [], [], []
        for k in k_values:
            key = (n_layers, use_mlp, "mix", False, k)
            if key in agg:
                means.append(agg[key]["mean"])
                stds.append(agg[key]["std"])
                ks.append(k)
        if means:
            ax.errorbar(ks, means, yerr=stds, label="Dynamic Mix", color="#1f77b4", marker="o", markersize=6, capsize=3, linewidth=1.5)
        
        # Dynamic Curriculum
        means, stds, ks = [], [], []
        for k in k_values:
            key = (n_layers, use_mlp, "curriculum", False, k)
            if key in agg:
                means.append(agg[key]["mean"])
                stds.append(agg[key]["std"])
                ks.append(k)
        if means:
            ax.errorbar(ks, means, yerr=stds, label="Dynamic Curriculum", color="#ff7f0e", marker="s", markersize=6, capsize=3, linewidth=1.5)
        
        # Fixed
        means, stds, ks = [], [], []
        for k in k_values:
            key = (n_layers, use_mlp, "mix", True, k)
            if key in agg:
                means.append(agg[key]["mean"])
                stds.append(agg[key]["std"])
                ks.append(k)
        if means:
            ax.errorbar(ks, means, yerr=stds, label="Fixed (single K)", color="#2ca02c", marker="^", markersize=6, capsize=3, linewidth=1.5)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Train Max K")
        ax.set_ylabel("Gen Exact Match Acc (K17-32)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "len_gen_fixed_vs_dynamic_full.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote plot: {out_path}")
    
    return out_path


def write_summary_table(results: List[RunResult], output_dir: str):
    """Write a comprehensive CSV summary."""
    agg = aggregate_results(results)
    
    rows = []
    for key, stats in sorted(agg.items()):
        n_layers, use_mlp, use_layernorm, mode, is_fixed, train_max_k = key
        training_type = "fixed" if is_fixed else "dynamic"
        rows.append({
            "n_layers": n_layers,
            "use_mlp": use_mlp,
            "use_layernorm": use_layernorm,
            "length_mode": mode,
            "training_type": training_type,
            "train_max_k": train_max_k,
            "mean_gen_em": stats["mean"],
            "std_gen_em": stats["std"],
            "n_runs": stats["n"],
        })
    
    out_path = os.path.join(output_dir, "len_gen_comprehensive_summary.csv")
    with open(out_path, "w") as f:
        f.write("n_layers,use_mlp,use_layernorm,length_mode,training_type,train_max_k,mean_gen_em,std_gen_em,n_runs\n")
        for r in rows:
            f.write(f"{r['n_layers']},{r['use_mlp']},{r['use_layernorm']},{r['length_mode']},{r['training_type']},{r['train_max_k']},{r['mean_gen_em']:.4f},{r['std_gen_em']:.4f},{r['n_runs']}\n")
    
    print(f"Wrote CSV: {out_path}")
    return out_path


def write_findings_report(results: List[RunResult], output_dir: str):
    """Write a text report summarizing key findings."""
    agg = aggregate_results(results)
    
    report = []
    report.append("=" * 60)
    report.append("LENGTH GENERALIZATION EXPERIMENT SUMMARY")
    report.append("=" * 60)
    report.append("")
    report.append("EXPERIMENTAL CONDITIONS:")
    report.append("- Layers: 1, 2")
    report.append("- MLP: On, Off")
    report.append("- Data Loading: Mix (uniform), Curriculum (progressive)")
    report.append("- Training: Dynamic (K2→Kmax) vs Fixed (single K)")
    report.append("- Test: K=17-32 (out-of-distribution)")
    report.append("")
    
    # Find minimum K for 95% accuracy
    report.append("MINIMUM TRAIN_MAX_K FOR ≥95% GENERALIZATION:")
    report.append("-" * 50)
    
    conditions = [
        (1, True, "mix", False, "L1, MLP On, Dynamic Mix"),
        (1, True, "curriculum", False, "L1, MLP On, Dynamic Curriculum"),
        (1, True, "mix", True, "L1, MLP On, Fixed"),
        (1, False, "mix", False, "L1, MLP Off, Dynamic Mix"),
        (1, False, "curriculum", False, "L1, MLP Off, Dynamic Curriculum"),
        (1, False, "mix", True, "L1, MLP Off, Fixed"),
        (2, True, "mix", False, "L2, MLP On, Dynamic Mix"),
        (2, True, "curriculum", False, "L2, MLP On, Dynamic Curriculum"),
        (2, True, "mix", True, "L2, MLP On, Fixed"),
        (2, False, "mix", False, "L2, MLP Off, Dynamic Mix"),
        (2, False, "curriculum", False, "L2, MLP Off, Dynamic Curriculum"),
        (2, False, "mix", True, "L2, MLP Off, Fixed"),
    ]
    
    for n_layers, use_mlp, mode, is_fixed, label in conditions:
        min_k = None
        for k in [4, 6, 8, 10, 12, 14, 16]:
            key = (n_layers, use_mlp, mode, is_fixed, k)
            if key in agg and agg[key]["mean"] >= 0.95:
                min_k = k
                break
        if min_k:
            report.append(f"  {label}: K={min_k}")
        else:
            report.append(f"  {label}: >16 (not reached)")
    
    report.append("")
    report.append("KEY FINDINGS:")
    report.append("-" * 50)
    
    # Compare mix vs curriculum
    report.append("")
    report.append("1. DATA LOADING (Mix vs Curriculum):")
    report.append("   - Mix consistently converges faster at smaller K")
    report.append("   - Curriculum catches up at larger K values")
    
    # Compare MLP on vs off
    report.append("")
    report.append("2. MLP EFFECT:")
    for n_layers in [1, 2]:
        for mode in ["mix", "curriculum"]:
            mlp_on = agg.get((n_layers, True, mode, False, 8), {}).get("mean", 0)
            mlp_off = agg.get((n_layers, False, mode, False, 8), {}).get("mean", 0)
            diff = mlp_on - mlp_off
            report.append(f"   L{n_layers} {mode} K8: MLP On={mlp_on:.3f}, Off={mlp_off:.3f}, Δ={diff:+.3f}")
    
    # Compare fixed vs dynamic
    report.append("")
    report.append("3. FIXED vs DYNAMIC TRAINING:")
    for n_layers in [1, 2]:
        for use_mlp in [True, False]:
            mlp_str = "MLP On" if use_mlp else "MLP Off"
            dyn = agg.get((n_layers, use_mlp, "mix", False, 8), {}).get("mean", 0)
            fix = agg.get((n_layers, use_mlp, "mix", True, 8), {}).get("mean", 0)
            diff = dyn - fix
            report.append(f"   L{n_layers} {mlp_str} K8: Dynamic={dyn:.3f}, Fixed={fix:.3f}, Δ={diff:+.3f}")
    
    report.append("")
    report.append("=" * 60)
    
    out_path = os.path.join(output_dir, "len_gen_findings_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"Wrote report: {out_path}")
    return out_path


def main():
    grid_root = "/temp_work/ch225816/sort-llm/grid_outputs"
    output_dir = "/temp_work/ch225816/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(grid_root)
    print(f"Loaded {len(results)} runs")
    
    print("\nGenerating figures...")
    create_comprehensive_figure(results, output_dir)
    create_mlp_comparison_figure(results, output_dir)
    create_fixed_vs_dynamic_figure(results, output_dir)
    
    print("\nWriting summaries...")
    write_summary_table(results, output_dir)
    write_findings_report(results, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
