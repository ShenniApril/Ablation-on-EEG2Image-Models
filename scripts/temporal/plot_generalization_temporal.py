"""
plot_generalization_temporal.py
===============================
Generate the group-level temporal generalization figures with unified export
sizes and labels for the final Results section.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _candidate_site_packages() -> list[Path]:
    python_root = Path(sys.executable).resolve().parent
    candidates = [
        python_root / "Lib" / "site-packages",
        Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages",
        Path.home() / "anaconda3" / "Lib" / "site-packages",
        Path.home() / "anaconda3" / "envs" / "neurobridge" / "Lib" / "site-packages",
    ]
    return [path for path in candidates if path.exists()]


def _bootstrap_site_packages() -> None:
    for candidate in _candidate_site_packages():
        site_packages = str(candidate)
        if site_packages not in sys.path:
            sys.path.append(site_packages)


_bootstrap_site_packages()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if SCRIPT_DIR.name == "temporal_ablation":
    RESULTS_DIR = SCRIPT_DIR / "results" / "generalization_ablation"
    OUTPUT_DIR = REPO_ROOT / "assets" / "generalization_temporal"
else:
    RESULTS_DIR = REPO_ROOT / "results" / "temporal" / "generalization_ablation"
    OUTPUT_DIR = REPO_ROOT / "assets" / "temporal"

WIDTH_CM = 14.5
HEIGHT_STANDARD_CM = 8.8
HEIGHT_TALL_CM = 11.5
# 论文并排尺寸: figure* 内两个 minipage 各 0.49\textwidth ≈ 8.9cm,
# LaTeX 端按原始尺寸 1:1 放置, 字号保持真实 pt 值, 避免缩放后字变小。
# 窄版绘图区仅约 6cm, 故刻度标签与格内数值字号相应下调并旋转 x 轴标签。
PAPER_WIDTH_CM = 8.9
PAPER_HEIGHT_CM = 7.8
PAPER_TICK_LABEL_SIZE = 7.2
PAPER_CELL_TEXT_SIZE = 6.5
DPI = 600

AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.5
ASTERISK_SIZE = 10
FULL_PROFILE_TICK_SIZE = 7.4

matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    }
)

TW_ORDER = ["T0_0-50ms", "T1_50-150ms", "T2_150-300ms", "T3_300-500ms", "T4_500-800ms"]
FB_ORDER = ["delta", "theta", "alpha", "beta", "low_gamma", "gamma", "high_gamma"]
TW_SHORT_LABELS = {
    "T0_0-50ms": "T0 (0-52ms)",
    "T1_50-150ms": "T1 (52-152ms)",
    "T2_150-300ms": "T2 (150-300ms)",
    "T3_300-500ms": "T3 (300-500ms)",
    "T4_500-800ms": "T4 (500-800ms)",
}
FB_SHORT_LABELS = {
    "delta": "delta",
    "theta": "theta",
    "alpha": "alpha",
    "beta": "beta",
    "low_gamma": "low-gamma",
    "gamma": "gamma",
    "high_gamma": "high-gamma",
}

TOP1_COLOR = "#E74C3C"
TOP5_COLOR = "#3498DB"


def cm_to_inches(value_cm: float) -> float:
    return value_cm / 2.54


def figure_size(height_cm: float) -> tuple[float, float]:
    return (cm_to_inches(WIDTH_CM), cm_to_inches(height_cm))


def save_figure(fig: plt.Figure, out_file: Path) -> None:
    fig.savefig(out_file, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[Trace] output: {out_file}")


def short_setting_name(setting: str) -> str:
    return setting.replace("-subjects", "")


def short_time_label(time_window: str) -> str:
    return TW_SHORT_LABELS.get(time_window, time_window.split("_", 1)[0])


def short_freq_label(freq_band: str) -> str:
    return FB_SHORT_LABELS.get(freq_band, freq_band.replace("_", "-"))


def format_condition_label(name: str) -> str:
    if "__" not in name:
        return name
    time_window, freq_band = name.split("__", 1)
    return f"{short_time_label(time_window)} x {short_freq_label(freq_band)}"


def get_sig_asterisks(p_raw: float, p_fdr: float) -> str:
    if p_raw < 0.05 and p_fdr < 0.05:
        return "**"
    if p_raw < 0.05:
        return "*"
    return ""


def process_data(setting: str) -> tuple[pd.DataFrame | None, float | None]:
    all_data = []
    baseline_top1s = []
    baseline_by_subject = {}
    used_csvs = []

    for sub in range(1, 11):
        csv_path = RESULTS_DIR / f"generalization_stft_ablation_{setting}_sub{sub:02d}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["subject"] = sub
        all_data.append(df)
        used_csvs.append(csv_path)

        baseline_row = df[df["name"] == "baseline"]
        if not baseline_row.empty:
            baseline = float(baseline_row.iloc[0]["top1"])
            baseline_top1s.append(baseline)
            baseline_by_subject[sub] = baseline

    if not all_data:
        print(f"[Skip] No CSV files found for {setting}")
        return None, None

    print(f"[Trace] {setting}: loaded {len(used_csvs)} CSV files")
    full_df = pd.concat(all_data, ignore_index=True)
    plot_df = full_df[full_df["name"].str.contains("__", na=False)].copy()
    plot_df = plot_df[plot_df["subject"].isin(baseline_by_subject)].copy()
    plot_df["paired_top1_drop"] = plot_df.apply(
        lambda row: baseline_by_subject[int(row["subject"])] - float(row["top1"]),
        axis=1,
    )
    print("[Trace] Top-1 drops recomputed from paired subject accuracies")

    avg_baseline = float(np.mean(baseline_top1s)) if baseline_top1s else None
    stats_records = []

    for name, group in plot_df.groupby("name"):
        drops1 = group["paired_top1_drop"].dropna().astype(float).values
        drops5 = group["top5_drop"].dropna().astype(float).values
        if len(drops1) == 0:
            continue

        p_raw = 1.0
        if len(drops1) > 1:
            if np.std(drops1, ddof=1) == 0.0:
                p_raw = 1.0 if np.mean(drops1) == 0.0 else 0.0
            else:
                _, p_raw = ttest_1samp(drops1, 0)

        time_window, freq_band = name.split("__", 1)
        stats_records.append(
            {
                "name": name,
                "time_window": time_window,
                "freq_band": freq_band,
                "mean1": float(np.mean(drops1)),
                "sem1": float(np.std(drops1, ddof=1) / np.sqrt(len(drops1))) if len(drops1) > 1 else 0.0,
                "mean5": float(np.mean(drops5)) if len(drops5) > 0 else 0.0,
                "sem5": float(np.std(drops5, ddof=1) / np.sqrt(len(drops5))) if len(drops5) > 1 else 0.0,
                "n_subjects": int(len(drops1)),
                "p_raw": float(p_raw),
            }
        )

    stats_df = pd.DataFrame(stats_records)
    if stats_df.empty:
        print(f"[Skip] No valid time-frequency conditions found for {setting}")
        return None, avg_baseline

    _, p_fdrs, _, _ = multipletests(stats_df["p_raw"], method="fdr_bh")
    stats_df["p_fdr"] = p_fdrs
    stats_df["sig"] = stats_df.apply(lambda row: get_sig_asterisks(row["p_raw"], row["p_fdr"]), axis=1)
    return stats_df, avg_baseline


def plot_heatmap(stats_df: pd.DataFrame, setting: str, output_dir: Path) -> None:
    matrix = np.full((len(TW_ORDER), len(FB_ORDER)), np.nan)
    for row_idx, time_window in enumerate(TW_ORDER):
        for col_idx, freq_band in enumerate(FB_ORDER):
            row = stats_df[(stats_df["time_window"] == time_window) & (stats_df["freq_band"] == freq_band)]
            if not row.empty:
                matrix[row_idx, col_idx] = float(row.iloc[0]["mean1"])

    fig, ax = plt.subplots(
        figsize=(cm_to_inches(PAPER_WIDTH_CM), cm_to_inches(PAPER_HEIGHT_CM))
    )
    vmax = max(0.15, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 0.15
    vmin = min(0.0, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0.0
    image = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")

    for row_idx, time_window in enumerate(TW_ORDER):
        for col_idx, freq_band in enumerate(FB_ORDER):
            cond_name = f"{time_window}__{freq_band}"
            row = stats_df[stats_df["name"] == cond_name]
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                ax.text(col_idx, row_idx, "N/A", ha="center", va="center", fontsize=PAPER_CELL_TEXT_SIZE, color="gray")
                continue

            color = "white" if abs(value) > vmax * 0.6 else "black"
            # 并排布局下绘图区宽约 5cm, 7 列格宽不足以容纳 3 位小数 (6 字符),
            # 故论文版保留 2 位小数 (5 字符), 精确值见正文 Table
            text_value = f"{value:+.2f}"
            if text_value == "-0.00":
                text_value = "+0.00"
            if not row.empty and row.iloc[0]["sig"]:
                text_value += f"\n{row.iloc[0]['sig']}"
            ax.text(
                col_idx,
                row_idx,
                text_value,
                ha="center",
                va="center",
                fontsize=PAPER_CELL_TEXT_SIZE,
                color=color,
                fontweight="bold",
            )

    if not np.all(np.isnan(matrix)):
        for row_idx, time_window in enumerate(TW_ORDER):
            for col_idx, freq_band in enumerate(FB_ORDER):
                cond_name = f"{time_window}__{freq_band}"
                row = stats_df[stats_df["name"] == cond_name]
                if not row.empty and row.iloc[0]["sig"] == "**":
                    ax.add_patch(
                        plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6)
                    )

    ax.set_xticks(range(len(FB_ORDER)))
    ax.set_xticklabels(
        [short_freq_label(freq_band) for freq_band in FB_ORDER],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=PAPER_TICK_LABEL_SIZE,
    )
    ax.set_yticks(range(len(TW_ORDER)))
    ax.set_yticklabels(
        [short_time_label(time_window) for time_window in TW_ORDER],
        fontsize=PAPER_TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="y", pad=4)
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Time Window")

    # 显式指定 fraction, 避免 colorbar 默认按比例二次压缩本已很窄的绘图区
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Top-1 accuracy drop")
    colorbar.ax.tick_params(labelsize=PAPER_TICK_LABEL_SIZE)

    fig.subplots_adjust(left=0.215, right=0.84, bottom=0.25, top=0.96)
    save_figure(fig, output_dir / f"group_{short_setting_name(setting)}_temporal_heatmap.png")


def plot_top10_bar(stats_df: pd.DataFrame, setting: str, output_dir: Path) -> None:
    bar_df = stats_df.sort_values("mean1", ascending=False).head(10).copy()
    if bar_df.empty:
        print(f"[Skip] No top-10 data for {setting}")
        return

    names = [format_condition_label(name) for name in bar_df["name"]]
    top1 = [float(value) for value in bar_df["mean1"]]
    top5 = [float(value) for value in bar_df["mean5"]]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    ax.bar(x - width / 2, top1, width, label="Top-1 drop", color=TOP1_COLOR, alpha=0.88)
    ax.bar(x + width / 2, top5, width, label="Top-5 drop", color=TOP5_COLOR, alpha=0.88)

    for idx, row in enumerate(bar_df.reset_index(drop=True).itertuples()):
        if not row.sig:
            continue
        y_pos = top1[idx] + 0.01 if top1[idx] >= 0 else top1[idx] - 0.02
        ax.text(
            x[idx] - width / 2,
            y_pos,
            row.sig,
            ha="center",
            va="bottom" if top1[idx] >= 0 else "top",
            fontsize=ASTERISK_SIZE,
            fontweight="bold",
            color="black",
        )

    max_y = max(max(top1), max(top5))
    min_y = min(min(top1), min(top5), 0)
    ax.set_ylim(min_y * 1.15 if min_y < 0 else -0.01, max_y * 1.15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Accuracy drop")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.96)
    save_figure(fig, output_dir / f"group_{short_setting_name(setting)}_temporal_top10.png")


def plot_full_profile_bar(stats_df: pd.DataFrame, setting: str, output_dir: Path) -> None:
    bar_df = stats_df.sort_values("mean1", ascending=False).copy()
    if bar_df.empty:
        print(f"[Skip] No full-profile data for {setting}")
        return

    names = [format_condition_label(name) for name in bar_df["name"]]
    top1 = [float(value) for value in bar_df["mean1"]]
    top5 = [float(value) for value in bar_df["mean5"]]
    y = np.arange(len(names))
    height = 0.36

    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_TALL_CM))
    ax.barh(y - height / 2, top1, height, label="Top-1 drop", color=TOP1_COLOR, alpha=0.88)
    ax.barh(y + height / 2, top5, height, label="Top-5 drop", color=TOP5_COLOR, alpha=0.88)

    for idx, row in enumerate(bar_df.reset_index(drop=True).itertuples()):
        if not row.sig:
            continue
        x_pos = top1[idx] + 0.004 if top1[idx] >= 0 else top1[idx] - 0.008
        ax.text(
            x_pos,
            y[idx] - height / 2,
            row.sig,
            va="center",
            ha="left" if top1[idx] >= 0 else "right",
            fontsize=ASTERISK_SIZE,
            fontweight="bold",
            color="black",
        )

    min_x = min(top1 + top5 + [0.0])
    max_x = max(top1 + top5 + [0.0])
    ax.set_xlim(min_x * 1.18 if min_x < 0 else -0.01, max_x * 1.18 if max_x > 0 else 0.1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", labelsize=FULL_PROFILE_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy drop")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.08, top=0.98)
    save_figure(fig, output_dir / f"group_{short_setting_name(setting)}_temporal_ablation.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot group-level temporal generalization figures")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[Trace] input dir:", RESULTS_DIR)
    print("[Trace] output dir:", args.output_dir)

    for setting in ["intra-subjects", "inter-subjects"]:
        print(f"\n[Trace] start plotting for {setting}")
        stats_df, _avg_baseline = process_data(setting)
        if stats_df is None:
            continue

        plot_heatmap(stats_df, setting, args.output_dir)
        plot_top10_bar(stats_df, setting, args.output_dir)
        plot_full_profile_bar(stats_df, setting, args.output_dir)


if __name__ == "__main__":
    main()
