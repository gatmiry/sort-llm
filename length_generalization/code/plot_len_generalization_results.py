#!/usr/bin/env python3
"""Aggregate and visualize length-generalization results from log files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunKey:
    """Unique configuration key for a run (seed-agnostic)."""

    layer: int
    length_mode: str
    train_min_k: int
    train_max_k: int
    test_min_k: int
    test_max_k: int


@dataclass
class RunRecord:
    """Parsed metrics for a single run log."""

    log_path: str
    key: RunKey
    gen_exact_match_by_k: Dict[int, float]


RUN_NAME_RE = re.compile(
    r"_L(?P<layer>\d+).*_len(?P<mode>mix|curriculum)_trainK(?P<kmin>\d+)to(?P<kmax>\d+)"
    r"_testK(?P<tmin>\d+)to(?P<tmax>\d+)_"
)
RUN_LINE_RE = re.compile(r"View run (?P<run_name>bs\S+)_nodup1")
GEN_ACC_RE = re.compile(r"gen/K(?P<k>\d+)/exact_match_acc\s+(?P<acc>[0-9.]+)")


def iter_log_files(paths: Iterable[str]) -> Iterable[str]:
    """Yield log file paths from one or more glob patterns."""
    for pattern in paths:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                yield path


def parse_run_name(run_name: str) -> Optional[RunKey]:
    """Parse run configuration from the wandb run name."""
    match = RUN_NAME_RE.search(run_name)
    if match is None:
        return None
    return RunKey(
        layer=int(match.group("layer")),
        length_mode=str(match.group("mode")),
        train_min_k=int(match.group("kmin")),
        train_max_k=int(match.group("kmax")),
        test_min_k=int(match.group("tmin")),
        test_max_k=int(match.group("tmax")),
    )


def parse_log_file(path: str) -> Optional[RunRecord]:
    """Parse a log file into a RunRecord if possible."""
    run_name: Optional[str] = None
    gen_acc_by_k: Dict[int, float] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if run_name is None:
                run_match = RUN_LINE_RE.search(line)
                if run_match:
                    run_name = run_match.group("run_name")

            gen_match = GEN_ACC_RE.search(line)
            if gen_match:
                k = int(gen_match.group("k"))
                acc = float(gen_match.group("acc"))
                gen_acc_by_k[k] = acc

    if run_name is None or not gen_acc_by_k:
        return None

    key = parse_run_name(run_name)
    if key is None:
        return None

    return RunRecord(log_path=path, key=key, gen_exact_match_by_k=gen_acc_by_k)


def parse_config_yaml(path: str) -> Dict[str, str]:
    """Parse a subset of W&B config YAML without external dependencies."""
    desired_keys = {
        "length_mode",
        "n_layers",
        "train_min_k",
        "train_max_k",
        "test_min_k",
        "test_max_k",
    }
    values: Dict[str, str] = {}
    current_key: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                if line.endswith(":"):
                    key = line[:-1].strip()
                    current_key = key if key in desired_keys else None
                continue
            if current_key and "value:" in line:
                _, value = line.split("value:", 1)
                values[current_key] = value.strip().strip('"')
                current_key = None
    return values


def parse_wandb_run(run_files_dir: str) -> Optional[RunRecord]:
    """Parse wandb summary/config files into a RunRecord."""
    summary_path = os.path.join(run_files_dir, "wandb-summary.json")
    config_path = os.path.join(run_files_dir, "config.yaml")
    if not (os.path.isfile(summary_path) and os.path.isfile(config_path)):
        return None

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as handle:
        summary = json.load(handle)

    config = parse_config_yaml(config_path)
    try:
        key = RunKey(
            layer=int(config["n_layers"]),
            length_mode=str(config["length_mode"]),
            train_min_k=int(config["train_min_k"]),
            train_max_k=int(config["train_max_k"]),
            test_min_k=int(config["test_min_k"]),
            test_max_k=int(config["test_max_k"]),
        )
    except KeyError:
        return None

    gen_acc_by_k: Dict[int, float] = {}
    for k in range(key.test_min_k, key.test_max_k + 1):
        metric_key = f"gen/K{k}/exact_match_acc"
        if metric_key in summary:
            gen_acc_by_k[k] = float(summary[metric_key])

    if not gen_acc_by_k:
        return None

    return RunRecord(log_path=summary_path, key=key, gen_exact_match_by_k=gen_acc_by_k)


def mean_gen_acc(record: RunRecord, k_min: int = 17, k_max: int = 32) -> Optional[float]:
    """Compute mean generalization exact-match accuracy over a K range."""
    values = [
        record.gen_exact_match_by_k[k]
        for k in range(k_min, k_max + 1)
        if k in record.gen_exact_match_by_k
    ]
    if not values:
        return None
    return mean(values)


def aggregate_records(records: List[RunRecord]) -> Dict[RunKey, List[float]]:
    """Aggregate mean generalization accuracy per configuration."""
    grouped: Dict[RunKey, List[float]] = defaultdict(list)
    for record in records:
        acc = mean_gen_acc(record, k_min=record.key.test_min_k, k_max=record.key.test_max_k)
        if acc is None:
            continue
        grouped[record.key].append(acc)
    return grouped


def compute_thresholds(
    grouped: Dict[RunKey, List[float]],
    threshold: float,
) -> Dict[Tuple[int, str], Optional[int]]:
    """Find minimum train_max_k achieving threshold for each (layer, mode)."""
    results: Dict[Tuple[int, str], Optional[int]] = {}
    keys_by_group: Dict[Tuple[int, str], List[RunKey]] = defaultdict(list)
    for key in grouped:
        keys_by_group[(key.layer, key.length_mode)].append(key)

    for group, keys in keys_by_group.items():
        best: Optional[int] = None
        for key in sorted(keys, key=lambda k: k.train_max_k):
            accs = grouped[key]
            if not accs:
                continue
            avg_acc = mean(accs)
            if avg_acc >= threshold:
                best = key.train_max_k
                break
        results[group] = best
    return results


def write_csv(grouped: Dict[RunKey, List[float]], path: str) -> None:
    """Write aggregated results to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "length_mode",
                "train_min_k",
                "train_max_k",
                "test_min_k",
                "test_max_k",
                "mean_gen_exact_match",
                "std_gen_exact_match",
                "num_runs",
            ]
        )
        for key, values in sorted(grouped.items(), key=lambda kv: (kv[0].layer, kv[0].length_mode, kv[0].train_max_k)):
            avg = mean(values)
            std = stdev(values) if len(values) > 1 else 0.0
            writer.writerow(
                [
                    key.layer,
                    key.length_mode,
                    key.train_min_k,
                    key.train_max_k,
                    key.test_min_k,
                    key.test_max_k,
                    f"{avg:.4f}",
                    f"{std:.4f}",
                    len(values),
                ]
            )


def plot_results(grouped: Dict[RunKey, List[float]], output_path: str) -> None:
    """Generate summary plots for mean generalization accuracy."""
    layers = sorted({key.layer for key in grouped})
    modes = ["mix", "curriculum"]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        for mode in modes:
            xs: List[int] = []
            ys: List[float] = []
            yerrs: List[float] = []
            for key in sorted(grouped, key=lambda k: k.train_max_k):
                if key.layer != layer or key.length_mode != mode:
                    continue
                values = grouped[key]
                xs.append(key.train_max_k)
                ys.append(mean(values))
                yerrs.append(stdev(values) if len(values) > 1 else 0.0)

            if xs:
                ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=3, label=mode)

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Train max K")
        ax.set_xticks(sorted({key.train_max_k for key in grouped if key.layer == layer}))
        ax.grid(True, alpha=0.2)
        ax.legend()

    axes[0].set_ylabel("Mean gen exact-match acc (K17-32)")
    fig.suptitle("Length generalization vs training max K")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def plot_fixed_vs_dynamic(grouped: Dict[RunKey, List[float]], output_path: str) -> None:
    """Plot fixed-length vs dynamic-length training for mix (and dynamic for curriculum)."""
    layers = sorted({key.layer for key in grouped})
    modes = ["mix", "curriculum"]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        for mode in modes:
            fixed_xs: List[int] = []
            fixed_ys: List[float] = []
            fixed_errs: List[float] = []
            dyn_xs: List[int] = []
            dyn_ys: List[float] = []
            dyn_errs: List[float] = []

            for key in sorted(grouped, key=lambda k: k.train_max_k):
                if key.layer != layer or key.length_mode != mode:
                    continue
                values = grouped[key]
                avg = mean(values)
                err = stdev(values) if len(values) > 1 else 0.0
                if key.train_min_k == key.train_max_k:
                    fixed_xs.append(key.train_max_k)
                    fixed_ys.append(avg)
                    fixed_errs.append(err)
                if key.train_min_k == 2 and key.train_max_k >= 2:
                    dyn_xs.append(key.train_max_k)
                    dyn_ys.append(avg)
                    dyn_errs.append(err)

            if fixed_xs:
                ax.errorbar(
                    fixed_xs,
                    fixed_ys,
                    yerr=fixed_errs,
                    marker="o",
                    capsize=3,
                    label=f"{mode} fixed",
                )
            if dyn_xs:
                ax.errorbar(
                    dyn_xs,
                    dyn_ys,
                    yerr=dyn_errs,
                    marker="s",
                    capsize=3,
                    label=f"{mode} dynamic",
                )

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Train max K")
        ax.set_xticks(sorted({key.train_max_k for key in grouped if key.layer == layer}))
        ax.grid(True, alpha=0.2)
        ax.legend()

    axes[0].set_ylabel("Mean gen exact-match acc (K17-32)")
    fig.suptitle("Fixed vs dynamic length training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main() -> None:
    """Entry point for plotting results from log files."""
    parser = argparse.ArgumentParser(description="Plot length generalization results from log files.")
    parser.add_argument(
        "--log-glob",
        nargs="+",
        default=[
            "/temp_work/ch225816/logs/len_gen-*.out",
            "/temp_work/ch225816/logs_old/len_gen-*.out",
        ],
        help="Glob patterns for log files.",
    )
    parser.add_argument(
        "--wandb-glob",
        nargs="+",
        default=[
            "/temp_work/ch225816/sort-llm/grid_outputs/**/wandb-summary.json",
        ],
        help="Glob patterns for wandb summary files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/temp_work/ch225816/plots",
        help="Directory for plots and CSV output.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records: List[RunRecord] = []
    for pattern in args.wandb_glob:
        for summary_path in glob.glob(pattern, recursive=True):
            run_files_dir = os.path.dirname(summary_path)
            record = parse_wandb_run(run_files_dir)
            if record is not None:
                records.append(record)

    for path in iter_log_files(args.log_glob):
        record = parse_log_file(path)
        if record is not None:
            records.append(record)

    grouped = aggregate_records(records)
    if not grouped:
        raise SystemExit("No completed logs with gen/K* metrics found.")

    csv_path = os.path.join(args.output_dir, "len_gen_summary.csv")
    plot_path = os.path.join(args.output_dir, "len_gen_summary.png")
    fixed_plot_path = os.path.join(args.output_dir, "len_gen_fixed_vs_dynamic.png")
    write_csv(grouped, csv_path)
    plot_results(grouped, plot_path)
    plot_fixed_vs_dynamic(grouped, fixed_plot_path)

    thresholds = compute_thresholds(grouped, threshold=0.95)
    threshold_path = os.path.join(args.output_dir, "len_gen_thresholds.txt")
    with open(threshold_path, "w", encoding="utf-8") as handle:
        for (layer, mode), k in sorted(thresholds.items()):
            handle.write(f"layer={layer} mode={mode} min_train_max_k@0.95={k}\n")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote plot: {fixed_plot_path}")
    print(f"Wrote thresholds: {threshold_path}")


if __name__ == "__main__":
    main()
