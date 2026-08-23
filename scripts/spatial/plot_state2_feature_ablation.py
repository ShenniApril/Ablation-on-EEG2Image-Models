"""Plot State 4 feature-perturbation dose-response curves.

Solid lines show the target State 4 intervention. Dashed lines show the
matched random control. Error bands are +/- 1 SD across random seeds when
multiple runs are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"top1": "#ED3431", "top5": "#4472C4"}
LABELS = {"top1": "Top-1", "top5": "Top-5"}


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["top1_drop_pp"] = 100.0 * frame["top1_drop_vs_official"]
    frame["top5_drop_pp"] = 100.0 * frame["top5_drop_vs_official"]
    summary = (
        frame.groupby(
            ["condition", "control", "ratio", "gfp_scale"],
            dropna=False,
        )
        .agg(
            n=("seed", "size"),
            modified_fraction=("modified_fraction", "mean"),
            top1_mean=("top1_drop_pp", "mean"),
            top1_sd=("top1_drop_pp", "std"),
            top5_mean=("top5_drop_pp", "mean"),
            top5_sd=("top5_drop_pp", "std"),
        )
        .reset_index()
    )
    return summary


def panel_data(
    summary: pd.DataFrame,
    condition: str,
    control_name: str,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    subset = summary[summary["condition"] == condition].copy()
    dose_column = "gfp_scale" if condition == "amplitude" else "ratio"
    doses = np.sort(subset[dose_column].dropna().unique())

    # Add the mathematically exact no-perturbation reference.
    baseline_dose = 1.0 if condition == "amplitude" else 0.0
    doses = np.unique(np.r_[doses, baseline_dose])

    series: dict[str, dict[str, np.ndarray]] = {}
    for control, key in (("none", "target"), (control_name, "control")):
        means = {"top1": [], "top5": []}
        sds = {"top1": [], "top5": []}
        for dose in doses:
            if dose == baseline_dose:
                means["top1"].append(0.0)
                means["top5"].append(0.0)
                sds["top1"].append(0.0)
                sds["top5"].append(0.0)
                continue
            row = subset[
                (subset["control"] == control)
                & np.isclose(subset[dose_column], dose)
            ]
            for metric in ("top1", "top5"):
                if row.empty:
                    means[metric].append(np.nan)
                    sds[metric].append(np.nan)
                else:
                    means[metric].append(float(row[f"{metric}_mean"].iloc[0]))
                    sd = float(row[f"{metric}_sd"].iloc[0])
                    sds[metric].append(0.0 if np.isnan(sd) else sd)
        series[key] = {
            "top1_mean": np.asarray(means["top1"]),
            "top1_sd": np.asarray(sds["top1"]),
            "top5_mean": np.asarray(means["top5"]),
            "top5_sd": np.asarray(sds["top5"]),
        }
    return doses, series


def plot_panel(
    axis: plt.Axes,
    doses: np.ndarray,
    series: dict[str, dict[str, np.ndarray]],
    title: str,
    xlabel: str,
) -> None:
    x = 100.0 * doses
    for group, linestyle, alpha in (
        ("target", "-", 0.15),
        ("control", "--", 0.10),
    ):
        for metric, marker in (("top1", "o"), ("top5", "s")):
            mean = series[group][f"{metric}_mean"]
            sd = series[group][f"{metric}_sd"]
            label = f"{LABELS[metric]}" if group == "target" else f"Modified {LABELS[metric]}"
            axis.plot(
                x,
                mean,
                color=COLORS[metric],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5,
                label=label,
            )
            valid = np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
            if np.any(valid):
                axis.fill_between(
                    x,
                    mean - sd,
                    mean + sd,
                    where=valid,
                    color=COLORS[metric],
                    alpha=alpha,
                    linewidth=0,
                )
    axis.axhline(0.0, color="#666666", linewidth=0.9)
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xticks(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results_sub01/microstate_feature/"
            "microstate_random_alternative_ablation_results.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_sub01/microstate_feature/"
            "state2_feature_dose_response.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = data[data["state"] == 2].copy()
    summary = summarize(data)

    configurations = [
        ("duration", "random-segment", "Duration", "Duration shortening (%)"),
        ("occurrence", "random-segment", "Occurrence", "Episodes removed (%)"),
        # ("amplitude", "random-time", "GFP amplitude", "GFP amplitude retained (%)"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.2),
        sharey=True,
        constrained_layout=False,
    )
    for axis, (condition, control, title, xlabel) in zip(axes, configurations):
        doses, series = panel_data(summary, condition, control)
        plot_panel(axis, doses, series, title, xlabel)

    axes[0].set_ylabel("Accuracy drop (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "STATE 2 Feature Perturbation",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.76, wspace=0.14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = args.output.with_name("state2_feature_ablation_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.output.with_suffix('.pdf')}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
