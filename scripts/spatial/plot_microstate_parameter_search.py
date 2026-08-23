"""Plot diagnostics from batch_microstate_clustering.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def short_label(row: pd.Series) -> str:
    return (
        f"T{int(row.fit_trials / 1000)}k/P{int(row.max_peaks)}/"
        f"D{row.peak_distance_ms:g}/S{row.min_segment_ms:g}/"
        f"I{int(row.n_init)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "results/microstate_parameter_search_k4/"
            "clustering_experiment_summary.csv"
        ),
    )
    parser.add_argument("--top-labels", type=int, default=3)
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    data = data[data["status"].eq("complete")].copy()
    numeric = [
        "test_gev",
        "test_mean_abs_r",
        "min_mean_coverage",
        "min_trial_presence",
        "max_template_abs_correlation",
        "fit_trials",
        "max_peaks",
        "peak_distance_ms",
        "min_segment_ms",
        "n_init",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric)
    data["label"] = data.apply(short_label, axis=1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
    colors = {10.0: "#3973c6", 20.0: "#ee7b2c"}
    markers = {10.0: "o", 20.0: "s"}

    for (distance, segment), group in data.groupby(
        ["peak_distance_ms", "min_segment_ms"]
    ):
        legend = f"peak {distance:g} ms · segment {segment:g} ms"
        axes[0].scatter(
            100 * group["min_mean_coverage"],
            100 * group["test_gev"],
            s=42 + 18 * (group["max_peaks"] - group["max_peaks"].min()),
            marker=markers.get(float(segment), "o"),
            color=colors.get(float(distance), "#666666"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.7,
            label=legend,
        )
        axes[1].scatter(
            group["max_template_abs_correlation"],
            100 * group["test_gev"],
            s=42 + 18 * (group["max_peaks"] - group["max_peaks"].min()),
            marker=markers.get(float(segment), "o"),
            color=colors.get(float(distance), "#666666"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.7,
        )

    best = data.nlargest(args.top_labels, "test_gev")
    annotation_offsets = [(7, 8), (7, -15), (-34, 8), (-34, -15)]
    for rank, ((_, row), offset) in enumerate(
        zip(best.iterrows(), annotation_offsets), start=1
    ):
        axes[0].annotate(
            f"#{rank}",
            (100 * row["min_mean_coverage"], 100 * row["test_gev"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    top = best.iloc[0]
    axes[0].text(
        0.02,
        0.98,
        f"#1: {top['label']}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )

    axes[0].set_title("Fit quality vs weakest-state coverage")
    axes[0].set_xlabel("Minimum mean coverage across states (%)")
    axes[0].set_ylabel("Test GEV (%)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")

    axes[1].set_title("Fit quality vs template redundancy")
    axes[1].set_xlabel("Maximum pairwise |spatial correlation|")
    axes[1].set_ylabel("Test GEV (%)")
    axes[1].axvline(0.90, color="#777777", linestyle=":", linewidth=1)
    axes[1].text(
        0.895,
        axes[1].get_ylim()[0],
        " redundancy warning",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=7.5,
        color="#666666",
    )

    factors = [
        ("fit_trials", "Fit trials"),
        ("max_peaks", "Max peaks"),
        ("peak_distance_ms", "Peak distance"),
        ("min_segment_ms", "Min segment"),
        ("n_init", "Initializations"),
    ]
    positions = np.arange(len(factors))
    for factor_index, (column, label) in enumerate(factors):
        grouped = (
            data.groupby(column)["test_gev"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values(column)
        )
        offsets = np.linspace(-0.16, 0.16, len(grouped))
        for offset, (_, row) in zip(offsets, grouped.iterrows()):
            x = factor_index + offset
            axes[2].plot(
                [x, x],
                [100 * row["min"], 100 * row["max"]],
                color="#999999",
                linewidth=1.5,
                zorder=1,
            )
            axes[2].scatter(
                x,
                100 * row["mean"],
                color="#3973c6",
                s=45,
                zorder=2,
            )
            value = row[column]
            value_text = (
                f"{int(value / 1000)}k"
                if column == "fit_trials"
                else f"{value:g}"
            )
            axes[2].annotate(
                value_text,
                (x, 100 * row["mean"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axes[2].set_xticks(positions, [label for _, label in factors], rotation=24)
    axes[2].set_ylabel("Mean test GEV across other settings (%)")
    axes[2].set_title("Marginal parameter effects")
    axes[2].text(
        0.02,
        0.02,
        "Dots: mean · vertical lines: observed min–max",
        transform=axes[2].transAxes,
        fontsize=8,
        color="#666666",
    )

    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"K=4 microstate parameter search ({len(data)} completed runs)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_png = args.csv.with_name("clustering_parameter_search.png")
    output_pdf = args.csv.with_name("clustering_parameter_search.pdf")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
