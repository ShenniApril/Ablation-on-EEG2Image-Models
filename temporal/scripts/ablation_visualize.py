"""
ablation_visualize.py
=====================
Generate the temporal ablation figures with a unified export standard for the
final Results section.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def _candidate_site_packages() -> list[Path]:
    python_root = Path(sys.executable).resolve().parent
    candidates = [
        python_root / "Lib" / "site-packages",
        Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages",
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


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_CANONICAL_DIR = SCRIPT_DIR / "results" / "temporal_sub08_phase1_canonical"
PHASE1_CSV = PHASE1_CANONICAL_DIR / "stft_ablation_results.csv"
PHASE1_MCNEMAR_CSV = PHASE1_CANONICAL_DIR / "mcnemar_fdr_results.csv"

PHASE2_CANONICAL_DIR = SCRIPT_DIR / "results" / "temporal_sub08_phase2_canonical" / "intra-subjects-averaged"
PHASE2_CSV = PHASE2_CANONICAL_DIR / "phase2_canonical_averaged.csv"
PHASE3_CSV = SCRIPT_DIR / "results" / "phase3_full_freq_masking" / "phase3_window_masking_results.csv"
CANONICAL_MCNEMAR_CSV = PHASE2_CANONICAL_DIR / "mcnemar_fdr_results.csv"
MCNEMAR_CSV = CANONICAL_MCNEMAR_CSV if CANONICAL_MCNEMAR_CSV.exists() else None
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "assets" / "temporal_ablation"

WIDTH_CM = 14.5
HEIGHT_STANDARD_CM = 8.8
HEIGHT_MEDIUM_CM = 10.2
HEIGHT_TALL_CM = 11.5
DPI = 600
IEEE_CONFERENCE_TEXTWIDTH_IN = 7.125
SUMMARY_HALF_COLUMN_WIDTH_RATIO = 0.48
IEEE_NORMAL_TEXT_PT = 10.0
SUMMARY_TARGET_BODY_RATIO = 0.80
SUMMARY_AXIS_FONT_SCALE = 1.05
SUMMARY_TICK_FONT_SCALE = 1.00
SUMMARY_LEGEND_FONT_SCALE = 1.00
SUMMARY_ASTERISK_FONT_SCALE = 1.15

AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.5
CELL_TEXT_SIZE = 8
ASTERISK_SIZE = 10

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
    "T2_150-300ms": "T2 (152-300ms)",
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
COLORS = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71", "#9B59B6", "#1ABC9C"]


def cm_to_inches(value_cm: float) -> float:
    return value_cm / 2.54


def figure_size(height_cm: float) -> tuple[float, float]:
    return (cm_to_inches(WIDTH_CM), cm_to_inches(height_cm))


def custom_figure_size(width_cm: float, height_cm: float) -> tuple[float, float]:
    return (cm_to_inches(width_cm), cm_to_inches(height_cm))


def compute_summary_font_config(args: argparse.Namespace) -> dict[str, float]:
    export_width_in = cm_to_inches(args.summary_export_width_cm)
    final_width_in = args.summary_final_width_ratio * args.summary_textwidth_in
    display_scale = final_width_in / export_width_in
    if display_scale <= 0:
        raise ValueError("Summary display scale must be positive.")

    target_final_font_pt = args.summary_body_font_pt * args.summary_target_body_ratio
    base_source_font_pt = target_final_font_pt / display_scale
    return {
        "export_width_cm": args.summary_export_width_cm,
        "export_width_in": export_width_in,
        "final_width_in": final_width_in,
        "display_scale": display_scale,
        "target_final_font_pt": target_final_font_pt,
        "base_source_font_pt": base_source_font_pt,
        "axis": base_source_font_pt * args.summary_axis_font_scale,
        "tick": base_source_font_pt * args.summary_tick_font_scale,
        "legend": base_source_font_pt * args.summary_legend_font_scale,
        "asterisk": base_source_font_pt * args.summary_asterisk_font_scale,
    }


def save_figure(fig: plt.Figure, out_path: Path | list[Path] | tuple[Path, ...]) -> None:
    out_paths = [out_path] if isinstance(out_path, Path) else list(out_path)
    for path in out_paths:
        fig.savefig(path, dpi=DPI, facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


def short_time_label(time_window: str) -> str:
    return TW_SHORT_LABELS.get(time_window, time_window.split("_", 1)[0])


def short_freq_label(freq_band: str) -> str:
    return FB_SHORT_LABELS.get(freq_band, freq_band.replace("_", "-"))


def format_condition_label(name: str) -> str:
    if name.startswith("full_freq_"):
        return short_time_label(name.replace("full_freq_", "", 1))
    if "__" in name:
        time_window, freq_band = name.split("__", 1)
        return f"{short_time_label(time_window)} x {short_freq_label(freq_band)}"
    return name


def summary_freq_label(freq_band: str) -> str:
    if freq_band == "alpha":
        return "α"
    if freq_band == "beta":
        return "β"
    return short_freq_label(freq_band)


def format_summary_condition_label(name: str) -> str:
    if "__" not in name:
        return name
    time_window, freq_band = name.split("__", 1)
    return f"{time_window.split('_', 1)[0]} x {summary_freq_label(freq_band)}"


def load_phase1(path: Path) -> tuple[dict[tuple[str, str], float], float | None]:
    if not path.exists():
        return {}, None

    data: dict[tuple[str, str], float] = {}
    baseline = None
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["name"] == "baseline":
                baseline = float(row["top1"])
                continue

            time_window = row.get("time_window", "")
            freq_band = row.get("freq_band", "")
            top1_drop = row.get("top1_drop", "")
            if time_window and freq_band and top1_drop not in ("", "None", None):
                try:
                    data[(time_window, freq_band)] = float(top1_drop)
                except ValueError:
                    pass
    return data, baseline


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_phase2_seed_summary_stats(phase2_csv_path: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    seeds_dir = phase2_csv_path.parent / "seeds"
    if not seeds_dir.exists():
        return {}

    grouped: dict[tuple[str, str, float], list[float]] = {}
    for seed_dir in sorted(path for path in seeds_dir.iterdir() if path.is_dir()):
        seed_csv = seed_dir / "phase2_seed_results.csv"
        if not seed_csv.exists():
            continue
        with seed_csv.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                condition = row.get("condition", "")
                perturbation = row.get("perturbation", "")
                param_value = row.get("param_value", "")
                top1_drop = row.get("top1_drop", "")
                if not condition or not perturbation or param_value in ("", None) or top1_drop in ("", "None", None):
                    continue
                try:
                    key = (condition, perturbation, float(param_value))
                    grouped.setdefault(key, []).append(float(top1_drop))
                except ValueError:
                    continue

    stats: dict[tuple[str, str, float], dict[str, float]] = {}
    for key, values in grouped.items():
        if not values:
            continue
        values_array = np.asarray(values, dtype=float)
        mean = float(values_array.mean())
        sem = float(values_array.std(ddof=1) / np.sqrt(len(values_array))) if len(values_array) > 1 else 0.0
        stats[key] = {"mean": mean, "sem": sem, "n": float(len(values_array))}
    return stats


def load_mcnemar_fdr(path: Path | None) -> dict[str, dict[str, float]]:
    if not path or not path.exists():
        return {}

    sig_map: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cond = row.get("Condition", "")
            if not cond:
                continue
            try:
                sig_map[cond] = {
                    "p_raw": float(row.get("p_value_raw", 1.0)),
                    "p_fdr": float(row.get("p_value_fdr", 1.0)),
                }
            except ValueError:
                continue
    return sig_map


def get_sig_asterisks(cond_info: dict[str, float] | None) -> str:
    if not cond_info:
        return ""
    p_raw = cond_info.get("p_raw", 1.0)
    p_fdr = cond_info.get("p_fdr", 1.0)
    if p_raw < 0.05 and p_fdr < 0.05:
        return "**"
    if p_raw < 0.05:
        return "*"
    return ""


def plot_heatmap(
    data: dict[tuple[str, str], float],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    matrix = np.full((len(TW_ORDER), len(FB_ORDER)), np.nan)
    for row_idx, time_window in enumerate(TW_ORDER):
        for col_idx, freq_band in enumerate(FB_ORDER):
            if (time_window, freq_band) in data:
                matrix[row_idx, col_idx] = data[(time_window, freq_band)]

    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_MEDIUM_CM))
    vmax = max(0.15, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 0.15
    vmin = min(0.0, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0.0
    image = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")

    for row_idx, time_window in enumerate(TW_ORDER):
        for col_idx, freq_band in enumerate(FB_ORDER):
            value = matrix[row_idx, col_idx]
            cond_name = f"{time_window}__{freq_band}"
            if np.isnan(value):
                ax.text(col_idx, row_idx, "N/A", ha="center", va="center", fontsize=CELL_TEXT_SIZE, color="gray")
                continue

            text_color = "white" if abs(value) > vmax * 0.6 else "black"
            ast = get_sig_asterisks(sig_map.get(cond_name))
            text_value = f"{value:+.3f}"
            if ast:
                text_value += f"\n{ast}"
            
            ax.text(
                col_idx,
                row_idx,
                text_value,
                ha="center",
                va="center",
                fontsize=CELL_TEXT_SIZE,
                color=text_color,
                fontweight="bold",
            )

    if not np.all(np.isnan(matrix)):
        for row_idx, time_window in enumerate(TW_ORDER):
            for col_idx, freq_band in enumerate(FB_ORDER):
                cond_name = f"{time_window}__{freq_band}"
                ast = get_sig_asterisks(sig_map.get(cond_name))
                if ast == "**":
                    ax.add_patch(
                        plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.2)
                    )

    ax.set_xticks(range(len(FB_ORDER)))
    ax.set_xticklabels([short_freq_label(freq_band) for freq_band in FB_ORDER])
    ax.set_yticks(range(len(TW_ORDER)))
    ax.set_yticklabels([short_time_label(time_window) for time_window in TW_ORDER])
    ax.tick_params(axis="y", pad=6)
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Time Window")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    colorbar.set_label("Top-1 accuracy drop")
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    fig.subplots_adjust(left=0.23, right=0.93, bottom=0.16, top=0.96)
    save_figure(fig, out_dir / "fig2_phase2_heatmap.png")


def plot_phase1_bar(
    rows: list[dict[str, str]],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    valid_rows = [
        row
        for row in rows
        if row.get("name") not in ("baseline", "full_mask_all", "random_control", "")
        and row.get("top1_drop") not in ("", "None", None)
    ]
    if not valid_rows:
        print("  [Skip Fig3] no data")
        return

    valid_rows.sort(key=lambda row: float(row["top1_drop"]), reverse=True)
    valid_rows = valid_rows[:10]
    names = [format_condition_label(row["name"]) for row in valid_rows]
    top1 = [float(row["top1_drop"]) for row in valid_rows]
    top5 = [float(row.get("top5_drop") or 0) for row in valid_rows]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    ax.bar(x - width / 2, top1, width, label="Top-1 drop", color="#E74C3C", alpha=0.88)
    ax.bar(x + width / 2, top5, width, label="Top-5 drop", color="#3498DB", alpha=0.88)

    for idx, row in enumerate(valid_rows):
        ast = get_sig_asterisks(sig_map.get(row["name"]))
        if not ast:
            continue
        y_pos = top1[idx] + 0.01 if top1[idx] >= 0 else top1[idx] - 0.02
        ax.text(
            x[idx] - width / 2,
            y_pos,
            ast,
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
    save_figure(fig, out_dir / "fig3_phase2_top10.png")


def generate_gamma_anomaly_report(rows: list[dict[str, str]], out_dir: Path) -> None:
    gamma_rows = [
        row
        for row in rows
        if "gamma" in row.get("freq_band", "").lower() and row.get("top1_drop") not in ("", "None", None)
    ]
    anomalies = []
    for row in gamma_rows:
        try:
            if float(row["top1_drop"]) < 0:
                anomalies.append(row)
        except ValueError:
            continue

    anomalies.sort(key=lambda row: float(row["top1_drop"]))

    report_lines = [
        "=== Gamma 频段反常现象分析报告 ===",
        "",
        "【现象概述】",
        "在时频消融实验中，通常预期掩码某一频段会导致模型解码性能下降（Accuracy Drop > 0）。",
        "但部分 Gamma 条件出现 Accuracy Drop < 0，提示该频段在部分时间窗中更接近噪声或干扰成分。",
        "",
        "【反常数值区间与频段特征】",
    ]
    for row in anomalies:
        report_lines.append(
            f"- 时间窗: {row['time_window']:<15} | 频段: {row['freq_band']:<12} | Top-1 Drop: {float(row['top1_drop']):.4f}"
        )

    report_lines.extend(
        [
            "",
            "【结论】",
            "Gamma 频段并非模型始终依赖的核心解码信息载体；在特定时间段消除它，可能表现为有效去噪。",
            "",
        ]
    )

    report_path = out_dir / "gamma_anomaly_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Saved report: {report_path}")


def plot_scaling_curve(
    rows: list[dict[str, str]],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    data = [row for row in rows if row.get("perturbation") == "scaling"]
    if not data:
        print("  [Skip Fig4] no scaling data")
        return

    conditions = sorted(set(row["condition"] for row in data))
    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    for idx, condition in enumerate(conditions):
        subset = sorted(
            [row for row in data if row["condition"] == condition],
            key=lambda row: float(row["param_value"]),
        )
        xs = [float(row["param_value"]) for row in subset]
        ys = [float(row["top1"]) for row in subset]
        color = COLORS[idx % len(COLORS)]
        ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=4.5, label=format_condition_label(condition))

        for point_idx, row in enumerate(subset):
            param_value = float(row["param_value"])
            if param_value == 1.0:
                continue
            cond_name = f"{condition}_scaling_alpha={param_value}"
            ast = get_sig_asterisks(sig_map.get(cond_name))
            if ast:
                ax.text(
                    xs[point_idx],
                    ys[point_idx] + 0.01,
                    ast,
                    ha="center",
                    va="bottom",
                    fontsize=ASTERISK_SIZE,
                    fontweight="bold",
                    color=color,
                )

    baseline_values = [float(row["top1"]) for row in data if float(row["param_value"]) == 1.0]
    if baseline_values:
        ax.axhline(np.mean(baseline_values), color="gray", linestyle="--", linewidth=1.2, label="Baseline")
    ax.axvspan(0.6, 0.8, alpha=0.09, color="orange")
    ax.set_xlim(-0.05, 2.1)
    ax.set_xlabel("Amplitude scale factor (alpha)")
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.96)
    save_figure(
        fig,
        [
            out_dir / "fig4_phase3a_scaling.png",
            out_dir / "fig4a_phase2_scaling.png",
        ],
    )


def plot_phase_curve(
    rows: list[dict[str, str]],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    data = [row for row in rows if row.get("perturbation") == "phase_rand"]
    if not data:
        print("  [Skip Fig5] no phase-randomization data")
        return

    conditions = sorted(set(row["condition"] for row in data))
    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    for idx, condition in enumerate(conditions):
        subset = sorted(
            [row for row in data if row["condition"] == condition],
            key=lambda row: float(row["param_value"]),
        )
        xs = [float(row["param_value"]) for row in subset]
        ys = [float(row["top1"]) for row in subset]
        color = COLORS[idx % len(COLORS)]
        ax.plot(xs, ys, "s-", color=color, linewidth=1.8, markersize=4.5, label=format_condition_label(condition))

        for point_idx, row in enumerate(subset):
            param_value = float(row["param_value"])
            if param_value == 0.0:
                continue
            cond_name = f"{condition}_phase_rand_rand_ratio={param_value}"
            ast = get_sig_asterisks(sig_map.get(cond_name))
            if ast:
                ax.text(
                    xs[point_idx],
                    ys[point_idx] + 0.01,
                    ast,
                    ha="center",
                    va="bottom",
                    fontsize=ASTERISK_SIZE,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Phase randomization ratio")
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.96)
    save_figure(
        fig,
        [
            out_dir / "fig5_phase3b_phase_rand.png",
            out_dir / "fig4b_phase2_phase_randomization.png",
        ],
    )


def plot_noise_curve(
    rows: list[dict[str, str]],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    data = [row for row in rows if row.get("perturbation") == "gaussian_noise"]
    if not data:
        print("  [Skip Fig6] no noise data")
        return

    conditions = sorted(set(row["condition"] for row in data))
    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    for idx, condition in enumerate(conditions):
        subset = sorted(
            [row for row in data if row["condition"] == condition],
            key=lambda row: float(row["param_value"]),
            reverse=True,
        )
        xs = [float(row["param_value"]) for row in subset]
        ys = [float(row["top1"]) for row in subset]
        color = COLORS[idx % len(COLORS)]
        ax.plot(xs, ys, "^-", color=color, linewidth=1.8, markersize=4.8, label=format_condition_label(condition))

        for point_idx, row in enumerate(subset):
            cond_name = f"{condition}_gaussian_noise_snr_db={float(row['param_value'])}"
            ast = get_sig_asterisks(sig_map.get(cond_name))
            if ast:
                ax.text(
                    xs[point_idx],
                    ys[point_idx] + 0.01,
                    ast,
                    ha="center",
                    va="bottom",
                    fontsize=ASTERISK_SIZE,
                    fontweight="bold",
                    color=color,
                )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.96)
    save_figure(
        fig,
        [
            out_dir / "fig6_phase3c_noise.png",
            out_dir / "fig4c_phase2_noise_injection.png",
        ],
    )


def plot_phase3_bar(
    rows: list[dict[str, str]],
    out_dir: Path,
    sig_map: dict[str, dict[str, float]] | None = None,
) -> None:
    sig_map = sig_map or {}
    valid_rows = [row for row in rows if row.get("name") not in ("baseline", "")]
    if not valid_rows:
        print("  [Skip Fig1] no full-frequency masking data")
        return

    valid_rows.sort(key=lambda row: TW_ORDER.index(row["time_window"]))
    names = [format_condition_label(row["name"]) for row in valid_rows]
    top1 = [float(row["top1_drop"]) for row in valid_rows]
    top5 = [float(row.get("top5_drop") or 0) for row in valid_rows]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figure_size(HEIGHT_STANDARD_CM))
    ax.bar(x - width / 2, top1, width, label="Top-1 drop", color="#E74C3C", alpha=0.88)
    ax.bar(x + width / 2, top5, width, label="Top-5 drop", color="#3498DB", alpha=0.88)

    for idx, row in enumerate(valid_rows):
        ast = get_sig_asterisks(sig_map.get("Phase3_" + row["name"]))
        if not ast:
            continue
        y_pos = top1[idx] + 0.01 if top1[idx] >= 0 else top1[idx] - 0.02
        ax.text(
            x[idx] - width / 2,
            y_pos,
            ast,
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
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy drop")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.96)
    save_figure(fig, out_dir / "fig1_phase1_full_freq.png")


def plot_phase2_summary(
    rows: list[dict[str, str]],
    out_dir: Path,
    seed_stats: dict[tuple[str, str, float], dict[str, float]] | None = None,
    sig_map: dict[str, dict[str, float]] | None = None,
    font_cfg: dict[str, float] | None = None,
) -> None:
    seed_stats = seed_stats or {}
    sig_map = sig_map or {}
    font_cfg = font_cfg or {
        "export_width_cm": WIDTH_CM,
        "axis": AXIS_LABEL_SIZE,
        "tick": TICK_LABEL_SIZE,
        "legend": LEGEND_SIZE,
        "asterisk": ASTERISK_SIZE,
    }
    conditions = sorted(set(row["condition"] for row in rows))
    if not conditions:
        print("  [Skip Fig7] no summary data")
        return

    perturbation_info = [
        ("scaling", 0.0, "Scaling (alpha=0.0)"),
        ("phase_rand", 1.0, "Phase rand. (ratio=1.0)"),
        ("gaussian_noise", -10.0, "Noise (SNR=-10 dB)"),
    ]

    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=custom_figure_size(font_cfg["export_width_cm"], HEIGHT_TALL_CM))

    all_drops: list[float] = []
    for bar_idx, (perturbation, endpoint, label) in enumerate(perturbation_info):
        drops = []
        sems = []
        for condition in conditions:
            matched = [
                row
                for row in rows
                if row["condition"] == condition
                and row["perturbation"] == perturbation
                and abs(float(row["param_value"]) - endpoint) < 0.01
            ]
            drop_value = float(matched[0]["top1_drop"]) if matched else 0.0
            drops.append(drop_value)
            all_drops.append(drop_value)
            summary_stats = seed_stats.get((condition, perturbation, endpoint))
            sem_value = summary_stats["sem"] if summary_stats else 0.0
            sems.append(sem_value)
            all_drops.extend([drop_value + sem_value, drop_value - sem_value])

            if not matched:
                continue
            if perturbation == "scaling":
                cond_name = f"{condition}_scaling_alpha={endpoint}"
            elif perturbation == "phase_rand":
                cond_name = f"{condition}_phase_rand_rand_ratio={endpoint}"
            else:
                cond_name = f"{condition}_gaussian_noise_snr_db={endpoint}"

            ast = get_sig_asterisks(sig_map.get(cond_name))
            if ast:
                x_pos = x[conditions.index(condition)] + (bar_idx - 1) * width
                y_offset = sem_value + 0.012
                y_pos = drop_value + y_offset if drop_value >= 0 else drop_value - y_offset
                ax.text(
                    x_pos,
                    y_pos,
                    ast,
                    ha="center",
                    va="bottom" if drop_value >= 0 else "top",
                    fontsize=font_cfg["asterisk"],
                    fontweight="bold",
                    color="black",
                )

        ax.bar(
            x + (bar_idx - 1) * width,
            drops,
            width,
            yerr=sems,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
            label=label,
            color=COLORS[bar_idx],
            alpha=0.88,
        )

    max_y = max(all_drops) if all_drops else 0
    min_y = min(all_drops + [0]) if all_drops else 0
    ax.set_ylim(min_y * 1.15 if min_y < 0 else -0.01, max_y * 1.15 if max_y != 0 else 0.1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([format_summary_condition_label(condition) for condition in conditions])
    ax.tick_params(axis="x", labelsize=font_cfg["tick"])
    ax.tick_params(axis="y", labelsize=font_cfg["tick"])
    ax.set_ylabel("Top-1 accuracy drop", fontsize=font_cfg["axis"])
    ax.legend(frameon=False, loc="upper right", fontsize=font_cfg["legend"])
    ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.96)
    save_figure(
        fig,
        [
            out_dir / "fig7_phase3_summary.png",
            out_dir / "fig4d_phase2_summary.png",
        ],
    )


def build_phase2_composite(out_dir: Path) -> None:
    panel_paths = [
        out_dir / "fig4a_phase2_scaling.png",
        out_dir / "fig4b_phase2_phase_randomization.png",
        out_dir / "fig4c_phase2_noise_injection.png",
        out_dir / "fig4d_phase2_summary.png",
    ]
    if not all(path.exists() for path in panel_paths):
        print("  [Skip Composite] missing one or more Fig.4A-D panels")
        return

    fig, axes = plt.subplots(2, 2, figsize=figure_size(HEIGHT_TALL_CM))
    for ax, path in zip(axes.flatten(), panel_paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.04, hspace=0.04)
    save_figure(fig, out_dir / "fig4_phase2_composite.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization for the temporal ablation figures")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 12, 123], default=123)
    parser.add_argument("--phase1-csv", type=Path, default=PHASE1_CSV)
    parser.add_argument("--phase2-csv", type=Path, default=PHASE2_CSV)
    parser.add_argument("--phase3-csv", type=Path, default=PHASE3_CSV)
    parser.add_argument("--mcnemar-csv", type=Path, default=MCNEMAR_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--summary-export-width-cm", type=float, default=WIDTH_CM)
    parser.add_argument("--summary-textwidth-in", type=float, default=IEEE_CONFERENCE_TEXTWIDTH_IN)
    parser.add_argument("--summary-final-width-ratio", type=float, default=SUMMARY_HALF_COLUMN_WIDTH_RATIO)
    parser.add_argument("--summary-body-font-pt", type=float, default=IEEE_NORMAL_TEXT_PT)
    parser.add_argument("--summary-target-body-ratio", type=float, default=SUMMARY_TARGET_BODY_RATIO)
    parser.add_argument("--summary-axis-font-scale", type=float, default=SUMMARY_AXIS_FONT_SCALE)
    parser.add_argument("--summary-tick-font-scale", type=float, default=SUMMARY_TICK_FONT_SCALE)
    parser.add_argument("--summary-legend-font-scale", type=float, default=SUMMARY_LEGEND_FONT_SCALE)
    parser.add_argument("--summary-asterisk-font-scale", type=float, default=SUMMARY_ASTERISK_FONT_SCALE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_font_cfg = compute_summary_font_config(args)
    print(f"[Visualize] Output dir: {args.output_dir}")
    if args.mcnemar_csv:
        print(f"[Visualize] Phase 2 canonical McNemar CSV: {args.mcnemar_csv}")
    else:
        print("[Visualize] Phase 2 canonical McNemar CSV not found; significance overlays disabled.")
    print(
        "[Visualize] Summary half-column font calibration: "
        f"export_width={summary_font_cfg['export_width_in']:.2f} in, "
        f"final_width={summary_font_cfg['final_width_in']:.2f} in, "
        f"display_scale={summary_font_cfg['display_scale']:.3f}, "
        f"target_final_font={summary_font_cfg['target_final_font_pt']:.2f} pt, "
        f"source_tick_font={summary_font_cfg['tick']:.2f} pt"
    )

    sig_map = load_mcnemar_fdr(args.mcnemar_csv) if args.mcnemar_csv else {}
    
    # Load Phase 1 canonical McNemar CSV if exists
    if PHASE1_MCNEMAR_CSV.exists():
        phase1_sig_map = load_mcnemar_fdr(PHASE1_MCNEMAR_CSV)
        sig_map.update(phase1_sig_map)
        print(f"[Visualize] Loaded Phase 1 FDR stats from {PHASE1_MCNEMAR_CSV}")

    if args.phase in (1, 12, 123):
        print("\n--- Temporal STFT figures ---")
        data, _baseline_top1 = load_phase1(args.phase1_csv)
        phase1_rows = load_csv_rows(args.phase1_csv)
        if data:
            plot_heatmap(data, args.output_dir, sig_map=sig_map)
        else:
            print("  [Skip Fig2] Phase 1 CSV not found or empty")
        if phase1_rows:
            plot_phase1_bar(phase1_rows, args.output_dir, sig_map=sig_map)
            generate_gamma_anomaly_report(phase1_rows, args.output_dir)

    if args.phase in (2, 12, 123):
        print("\n--- Perturbation figures ---")
        phase2_rows = load_csv_rows(args.phase2_csv)
        phase2_seed_stats = load_phase2_seed_summary_stats(args.phase2_csv)
        if phase2_rows:
            plot_scaling_curve(phase2_rows, args.output_dir, sig_map=sig_map)
            plot_phase_curve(phase2_rows, args.output_dir, sig_map=sig_map)
            plot_noise_curve(phase2_rows, args.output_dir, sig_map=sig_map)
            plot_phase2_summary(
                phase2_rows,
                args.output_dir,
                seed_stats=phase2_seed_stats,
                sig_map=sig_map,
                font_cfg=summary_font_cfg,
            )
            build_phase2_composite(args.output_dir)
        else:
            print("  [Skip Phase 2] Phase 2 CSV not found or empty")

    if args.phase in (3, 123):
        print("\n--- Full-frequency masking figure ---")
        phase3_rows = load_csv_rows(args.phase3_csv)
        if phase3_rows:
            plot_phase3_bar(phase3_rows, args.output_dir, sig_map=sig_map)
        else:
            print("  [Skip Fig1] Phase 3 CSV not found or empty")

    print(f"\n[OK] Unified temporal figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
