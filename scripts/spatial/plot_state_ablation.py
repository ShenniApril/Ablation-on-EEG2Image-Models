"""Plot microstate-specific effects after subtracting matched controls.

The plotted quantity is:

    target accuracy drop - mean matched-control accuracy drop

Positive values indicate that perturbing the target microstate was more
damaging than relocating an otherwise matched intervention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"top1": "#ED3131", "top5": "#4472C4"}


def prepare_results(
    target_path: Path,
    control_path: Path,
    control_name: str,
    smoothing_ms: float,
    error_kind: str,
) -> pd.DataFrame:
    target = pd.read_csv(target_path)
    control = pd.read_csv(control_path)

    target = target[
        target["condition"].eq("topography")
        & target["control"].eq("none")
        & np.isclose(target["interface_smoothing_ms"], smoothing_ms)
    ].copy()
    control = control[
        control["condition"].eq("topography")
        & control["control"].eq(control_name)
        & np.isclose(control["interface_smoothing_ms"], smoothing_ms)
    ].copy()
    if target.empty:
        raise ValueError("No matching target topography rows were found.")
    if control.empty:
        raise ValueError("No matching matched-control rows were found.")

    target_summary = (
        target.groupby("state", as_index=False)
        .agg(
            target_n=("seed", "size"),
            target_effective_fraction=(
                "effective_weighted_modified_fraction",
                "mean",
            ),
            target_top1_drop=("top1_drop_vs_official", "mean"),
            target_top5_drop=("top5_drop_vs_official", "mean"),
        )
    )
    control_summary = (
        control.groupby("state", as_index=False)
        .agg(
            control_n=("seed", "size"),
            control_effective_fraction=(
                "effective_weighted_modified_fraction",
                "mean",
            ),
            control_top1_drop_mean=("top1_drop_vs_official", "mean"),
            control_top1_drop_sd=("top1_drop_vs_official", "std"),
            control_top5_drop_mean=("top5_drop_vs_official", "mean"),
            control_top5_drop_sd=("top5_drop_vs_official", "std"),
        )
    )
    result = target_summary.merge(
        control_summary, on="state", how="inner", validate="one_to_one"
    )
    if len(result) != len(target_summary):
        raise ValueError("Some target states have no matched-control result.")

    mismatch = np.abs(
        result["target_effective_fraction"]
        - result["control_effective_fraction"]
    )
    if np.any(mismatch > 1e-6):
        states = result.loc[mismatch > 1e-6, "state"].tolist()
        raise ValueError(
            f"Effective perturbation fractions are not matched for states {states}."
        )

    for metric in ("top1", "top5"):
        result[f"{metric}_specific_effect"] = 100.0 * (
            result[f"target_{metric}_drop"]
            - result[f"control_{metric}_drop_mean"]
        )
        error = 100.0 * result[f"control_{metric}_drop_sd"].fillna(0.0)
        if error_kind == "sem":
            error = error / np.sqrt(result["control_n"].clip(lower=1))
        result[f"{metric}_error"] = error

    result["interface_smoothing_ms"] = smoothing_ms
    result["matched_control"] = control_name
    result["error_kind"] = error_kind
    return result.sort_values("state").reset_index(drop=True)


def plot_results(
    results: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    states = results["state"].astype(int).to_numpy()
    x = np.arange(len(states))
    width = 0.34

    fig, axis = plt.subplots(figsize=(8.8, 5.3))
    for offset, metric, label in (
        (-width / 2, "top1", "Modified Top-1"),
        (width / 2, "top5", "Modified Top-5"),
    ):
        values = results[f"{metric}_specific_effect"].to_numpy()
        errors = results[f"{metric}_error"].to_numpy()
        bars = axis.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=4,
            color=COLORS[metric],
            alpha=0.9,
            label=label,
        )
        for bar, value in zip(bars, values):
            vertical_offset = 3 if value >= 0 else -4
            va = "bottom" if value >= 0 else "top"
            axis.annotate(
                f"{value:+.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=9,
            )

    axis.axhline(0.0, color="#555555", linewidth=1)
    axis.set_xticks(x, [f"STATE {state}" for state in states])
    axis.set_ylabel(
        "Accuracy drop\n(percentage points)"
    )
    axis.set_title("STATES Ablation Effect On Accuracy", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    # axis.text(
    #     0.01,
    #     0.01,
    #     (
    #         f"Error bars: matched-control {results['error_kind'].iloc[0].upper()} "
    #         f"across seeds · smoothing "
    #         f"{results['interface_smoothing_ms'].iloc[0]:g} ms"
    #     ),
    #     transform=axis.transAxes,
    #     fontsize=8,
    #     color="#666666",
    # )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "results/microstate_ablation/"
            "microstate_random_alternative_ablation_results.csv"
        ),
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=Path(
            "results/microstate_ablation/"
            "microstate_random_alternative_ablation_results.csv"
        ),
    )
    parser.add_argument(
        "--control-name", default="random-segment"
    )
    parser.add_argument("--interface-smoothing-ms", type=float, default=0.0)
    parser.add_argument(
        "--error",
        choices=["sd", "sem"],
        default="sd",
        help="Error bars from matched-control seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/microstate_ablation/"
            "microstate_specific_effect.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    results = prepare_results(
        target_path=args.target,
        control_path=args.control,
        control_name=args.control_name,
        smoothing_ms=args.interface_smoothing_ms,
        error_kind=args.error,
    )
    plot_results(results, args.output, args.dpi)

    summary_path = args.output.with_name("microstate_specific_effect_summary.csv")
    results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.output.with_suffix('.pdf')}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
