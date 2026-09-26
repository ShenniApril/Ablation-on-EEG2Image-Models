#!/usr/bin/env python3
"""Recompute inter-subject temporal drops from subject-level CSVs.

Inputs are the existing five-seed-averaged subject CSVs. This script does not
rerun checkpoints; it pairs each subject's baseline and ablated Top-1 accuracy.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "results" / "temporal" / "generalization_ablation"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "temporal" / "generalization_inter_paired"
SUBJECTS = range(1, 11)


def read_subject(path: Path, subject: int) -> tuple[float, dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    baseline_rows = [row for row in rows if row.get("name") == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError(f"{path}: expected exactly one baseline row")
    condition_rows = {
        row["name"]: row for row in rows
        if row.get("name") and "__" in row["name"]
    }
    if len(condition_rows) != 35:
        raise ValueError(f"{path}: expected 35 time-frequency rows, got {len(condition_rows)}")
    baseline = float(baseline_rows[0]["top1"])
    return baseline, condition_rows


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values in original order."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted_sorted = [1.0] * count
    running_min = 1.0
    for rank_from_end in range(count - 1, -1, -1):
        original_index = order[rank_from_end]
        rank = rank_from_end + 1
        running_min = min(running_min, p_values[original_index] * count / rank)
        adjusted_sorted[rank_from_end] = min(1.0, running_min)
    adjusted = [1.0] * count
    for sorted_index, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[sorted_index]
    return adjusted


def fmt(value: float) -> str:
    return f"{value:.12g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    subject_data: dict[int, tuple[float, dict[str, dict[str, str]]]] = {}
    for subject in SUBJECTS:
        path = args.input_dir / f"generalization_stft_ablation_inter-subjects_sub{subject:02d}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        subject_data[subject] = read_subject(path, subject)

    common_conditions = set.intersection(
        *(set(data[1]) for data in subject_data.values())
    )
    if len(common_conditions) != 35:
        raise ValueError(f"Expected 35 common conditions, got {len(common_conditions)}")

    pair_rows: list[dict[str, str]] = []
    summary: list[dict[str, float | int | str]] = []
    for condition in sorted(common_conditions):
        drops: list[float] = []
        baselines: list[float] = []
        ablated: list[float] = []
        legacy_drops: list[float] = []
        for subject, (baseline, rows) in subject_data.items():
            row = rows[condition]
            perturbed = float(row["top1"])
            paired_drop = baseline - perturbed
            legacy = float(row["top1_drop"])
            baselines.append(baseline)
            ablated.append(perturbed)
            drops.append(paired_drop)
            legacy_drops.append(legacy)
            time_window, freq_band = condition.split("__", 1)
            pair_rows.append({
                "subject": str(subject),
                "condition": condition,
                "time_window": time_window,
                "freq_band": freq_band,
                "baseline_top1": fmt(baseline),
                "ablated_top1": fmt(perturbed),
                "recomputed_paired_drop": fmt(paired_drop),
                "stored_top1_drop": fmt(legacy),
                "recomputed_minus_stored": fmt(paired_drop - legacy),
                "source_csv": f"generalization_stft_ablation_inter-subjects_sub{subject:02d}.csv",
            })

        baseline_mean = float(np.mean(baselines))
        ablated_mean = float(np.mean(ablated))
        paired_mean = float(np.mean(drops))
        paired_sd = float(np.std(drops, ddof=1))
        if paired_sd == 0.0:
            t_statistic = 0.0 if paired_mean == 0.0 else math.copysign(math.inf, paired_mean)
            p_raw = 1.0 if paired_mean == 0.0 else 0.0
        else:
            test = ttest_1samp(drops, 0.0)
            t_statistic = float(test.statistic)
            p_raw = float(test.pvalue)
        time_window, freq_band = condition.split("__", 1)
        summary.append({
            "condition": condition,
            "time_window": time_window,
            "freq_band": freq_band,
            "n_subjects": len(drops),
            "mean_baseline_top1": baseline_mean,
            "mean_ablated_top1": ablated_mean,
            "mean_paired_drop": paired_mean,
            "sd_paired_drop": paired_sd,
            "sem_paired_drop": paired_sd / math.sqrt(len(drops)),
            "mean_identity_difference": paired_mean - (baseline_mean - ablated_mean),
            "mean_stored_top1_drop": float(np.mean(legacy_drops)),
            "max_abs_recomputed_vs_stored": float(max(abs(a - b) for a, b in zip(drops, legacy_drops))),
            "t_statistic": t_statistic,
            "p_raw": p_raw,
            "q_fdr": float("nan"),
            "significance": "",
        })

    q_values = bh_fdr([float(row["p_raw"]) for row in summary])
    for row, q_value in zip(summary, q_values):
        row["q_fdr"] = q_value
        if row["p_raw"] < 0.05 and q_value < 0.05:
            row["significance"] = "**"
        elif row["p_raw"] < 0.05:
            row["significance"] = "*"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "inter_paired_subject_drops.csv"
    summary_path = args.output_dir / "inter_paired_condition_summary.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                key: fmt(value) if isinstance(value, (float, np.floating)) else value
                for key, value in row.items()
            })

    max_legacy = max(float(row["max_abs_recomputed_vs_stored"]) for row in summary)
    max_identity = max(abs(float(row["mean_identity_difference"])) for row in summary)
    print(f"subjects={len(subject_data)} conditions={len(summary)} pair_rows={len(pair_rows)}")
    print(f"max_abs_recomputed_vs_stored={max_legacy:.12g}")
    print(f"max_abs_mean_identity_difference={max_identity:.12g}")
    print(f"FDR_significant={(sum(row['significance'] == '**' for row in summary))}/35")
    print(f"paired_rows={pairs_path}")
    print(f"condition_summary={summary_path}")


if __name__ == "__main__":
    main()
