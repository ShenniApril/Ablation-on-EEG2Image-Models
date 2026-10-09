"""
ablation_temporal_stft.py — Phase 1: STFT Time-Frequency Masking Ablation
==========================================================================
针对 NeuroBridge EEG-to-Image 解码模型的第一阶段 Temporal 消融实验。

核心逻辑：
  对指定时间窗内的指定频段施加 STFT 掩码（置零），其余部分保持不变，
  送入冻结的 NeuroBridge 模型推理，记录 Top-1/Top-5 准确率变化。

实验设计依据（B版本调查报告）：
  - 时间窗依据：Cichy et al. (2014) 解码峰值 102ms；Thorpe et al. (1996) 分化起点 152ms
  - 频段依据：Fries (2015) CTC 理论；FourierMask (Wang et al., 2026) 实验验证
  - 数据约束：THINGS-EEG2 (Gifford 2022) 采样率 250Hz，时间窗 0–250ms (250点)

重要注意：
  EEG 数据在 NeuroBridge 中已经过预处理，采样率为 250Hz，时间窗 0–250ms (250点)。
  时间窗单位为「采样点」（1点 = 4ms at 250Hz）。

用法示例（在本仓库根目录下运行；脚本会自动把仓库根注入 sys.path 以解析 module 包）：
  python ablation_temporal_stft.py
  python ablation_temporal_stft.py --subject 8 --device cuda
  python ablation_temporal_stft.py --mode priority   # 只跑优先级★5/★4
  python ablation_temporal_stft.py --mode full        # 跑全部组合

作者：乔钰成 / NEOschool 项目组
版本：2.0-B  (三档 Gamma: low_gamma 30-45Hz / gamma 45-70Hz / high_gamma 70-100Hz + Phase 3)
"""

from __future__ import annotations
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8','utf-8-sig'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import istft, stft
from torch.utils.data import DataLoader

# 注入仓库根路径，使 module.* 从本仓库解析（克隆后可直接运行）
REPO_ROOT = Path(__file__).resolve().parent.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.model import EEGProject
from module.projector import ProjectorLinear

# ===========================================================================
# 常量配置
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"

# NeuroBridge 使用的 17 个枕顶通道
SELECTED_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

# 采样率（THINGS-EEG2 预处理后为 250Hz）
FS = 250.0

# STFT 参数（基于报告 Section 2.4，针对 250Hz 采样率调整）
# nperseg=32 → 窗长 128ms @ 250Hz，频率分辨率 ≈ 7.8Hz
# noverlap=16 → 50% 重叠
STFT_NPERSEG = 32
STFT_NOVERLAP = 16
STFT_NFFT = 64  # 频率分辨率 ≈ 3.9Hz

# ---------------------------------------------------------------------------
# 时间窗定义（采样点，250Hz 下 1点=4ms）
# 依据：Cichy 2014 (102ms 峰值), Thorpe 1996 (152ms 分化), Gifford 2022 (解码时间曲线)
# ---------------------------------------------------------------------------
TIME_WINDOWS = {
    "T0_0-50ms":    (0,   13),   #   0–  52ms: V1 最早期响应（Cichy 2014: 起点48ms）
    "T1_50-150ms":  (13,  38),   #  52– 152ms: V1前馈峰值 (Cichy 2014: V1峰101ms)
    "T2_150-300ms": (38,  75),   # 152– 300ms: N170/类别解码峰值 (Thorpe 1996: 152-186ms)
    "T3_300-500ms": (75,  125),  # 300– 500ms: P300/注意力整合
    "T4_500-800ms": (125, 200),  # 500– 800ms: 晚期认知加工（超出数据范围取到200点/800ms等效边界）
    "T_full":       (0,   250),  # 全段对照（用于验证总体频率重要性）
}

# ---------------------------------------------------------------------------
# 频段定义（Hz）
# 依据：Fries 2015 (gamma前馈/beta反馈); FourierMask Wang 2026 (beta+gamma关键)
# 注意：250Hz 采样率下，Nyquist = 125Hz，hi_gamma 上限设为 120Hz
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 频段定义 v2-B
# 三档 Gamma：
#   low_gamma  30-45 Hz : 经典 Gamma 下端（Bhattacharyya 2019 BCI low-gamma）
#   gamma      45-70 Hz : 核心视觉 Gamma，Fries 2015 前馈峰值区域（V1/V2）
#   high_gamma 70-100 Hz: 宽带高 Gamma；100Hz = THINGS-EEG2 在线滤波上限
#                         注：70-100Hz scalp EEG EMG 伪迹主导，结果需谨慎
# 已删除 hi_gamma 80-120Hz：100-120Hz 超出数据集物理上限，全为数值噪声
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    "delta":      (1.0,   4.0),
    "theta":      (4.0,   8.0),
    "alpha":      (8.0,   13.0),
    "beta":       (13.0,  30.0),
    "low_gamma":  (30.0,  45.0),   # 经典 Gamma 下端
    "gamma":      (45.0,  70.0),   # 核心视觉 Gamma（Fries 2015）
    "high_gamma": (70.0, 100.0),   # 宽带 Gamma 高端（替代原 hi_gamma 80-120Hz）
}

# ---------------------------------------------------------------------------
# 优先级矩阵（来自 B 版本报告 Section 2.3）
# 5=最高, 4=高, 3=中, 2=低, 1=可选参考
# ---------------------------------------------------------------------------
# v2-B Priority Matrix (7 bands x 5 time windows)
PRIORITY_MATRIX = {
    # T0: 0-52ms
    ("T0_0-50ms",    "delta"):      1,
    ("T0_0-50ms",    "theta"):      1,
    ("T0_0-50ms",    "alpha"):      1,
    ("T0_0-50ms",    "beta"):       2,
    ("T0_0-50ms",    "low_gamma"):  2,
    ("T0_0-50ms",    "gamma"):      2,
    ("T0_0-50ms",    "high_gamma"): 2,
    # T1: 52-152ms (V1 feedforward peak 101ms, Cichy 2014)
    ("T1_50-150ms",  "delta"):      1,
    ("T1_50-150ms",  "theta"):      1,
    ("T1_50-150ms",  "alpha"):      2,
    ("T1_50-150ms",  "beta"):       3,
    ("T1_50-150ms",  "low_gamma"):  5,
    ("T1_50-150ms",  "gamma"):      5,
    ("T1_50-150ms",  "high_gamma"): 3,
    # T2: 152-300ms (N170 / categorisation, Thorpe 1996: onset 152ms)
    ("T2_150-300ms", "delta"):      1,
    ("T2_150-300ms", "theta"):      2,
    ("T2_150-300ms", "alpha"):      2,
    ("T2_150-300ms", "beta"):       4,
    ("T2_150-300ms", "low_gamma"):  5,
    ("T2_150-300ms", "gamma"):      5,
    ("T2_150-300ms", "high_gamma"): 3,
    # T3: 300-500ms (P300 / attentional integration)
    ("T3_300-500ms", "delta"):      1,
    ("T3_300-500ms", "theta"):      3,
    ("T3_300-500ms", "alpha"):      3,
    ("T3_300-500ms", "beta"):       2,
    ("T3_300-500ms", "low_gamma"):  2,
    ("T3_300-500ms", "gamma"):      2,
    ("T3_300-500ms", "high_gamma"): 2,
    # T4: 500-800ms (late cognition)
    ("T4_500-800ms", "delta"):      1,
    ("T4_500-800ms", "theta"):      2,
    ("T4_500-800ms", "alpha"):      2,
    ("T4_500-800ms", "beta"):       1,
    ("T4_500-800ms", "low_gamma"):  1,
    ("T4_500-800ms", "gamma"):      1,
    ("T4_500-800ms", "high_gamma"): 1,
}

# 特殊对照组（额外添加，不在优先级矩阵中）
CONTROL_CONDITIONS = [
    {"name": "baseline",       "time_window": None,          "freq_band": None},
    {"name": "full_mask_all",  "time_window": "T_full",      "freq_band": "all_bands"},  # 全部频段置零
    {"name": "random_control",   "time_window": "T1_50-150ms", "freq_band": "delta"},        # 随机低优先级对照
    {"name": "full_time_gamma", "time_window": "T_full",      "freq_band": "gamma"},          # 全段 gamma(45-70Hz)，验证时间特异性
]


# ===========================================================================
# 核心处理函数
# ===========================================================================

def stft_mask_time_freq(
    eeg: np.ndarray,
    time_window: tuple[int, int],
    freq_band: tuple[float, float],
    fs: float = FS,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    nfft: int = STFT_NFFT,
) -> np.ndarray:
    """
    在 EEG 信号的指定时间窗内，对指定频段施加 STFT 掩码（置零）。

    参数
    ----
    eeg         : shape (..., channels, time_points)，float32
    time_window : (start_sample, end_sample)，待掩码的时间窗（采样点）
    freq_band   : (low_hz, high_hz)，待掩码的频段（Hz）
    fs          : 采样率（Hz）
    nperseg     : STFT 窗长（采样点）
    noverlap    : STFT 重叠长度（采样点）
    nfft        : FFT 点数

    返回
    ----
    ablated_eeg : 与输入相同 shape，指定时间窗内指定频段已置零后重建

    原理
    ----
    1. 截取目标时间窗的 EEG 片段
    2. 对该片段做 STFT → 时频复数矩阵
    3. 将目标频段对应的 STFT 系数置零（振幅+相位同时清除）
    4. ISTFT 重建时域信号
    5. 将重建信号填回原始 EEG 的对应位置
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    original_shape = eeg.shape
    t_start, t_end = time_window
    low_hz, high_hz = freq_band

    # 将 (..., C, T) 展平为 (N, T) 方便批量处理
    flat = eeg.reshape(-1, eeg.shape[-1])
    ablated_flat = flat.copy()

    # 截取时间窗片段
    segment = flat[:, t_start:t_end]  # (N, seg_len)

    if segment.shape[-1] < nperseg:
        # 时间窗太短无法做 STFT，降级为频域直接置零（FFT→掩码→IFFT）
        fft_seg = np.fft.rfft(segment, axis=-1)
        freqs_fft = np.fft.rfftfreq(segment.shape[-1], d=1.0/fs)
        freq_mask_fft = (freqs_fft >= low_hz) & (freqs_fft <= high_hz)
        fft_seg[:, freq_mask_fft] = 0.0
        ablated_segment = np.fft.irfft(fft_seg, n=segment.shape[-1], axis=-1).astype(np.float32)
        ablated_flat[:, t_start:t_end] = ablated_segment
        return ablated_flat.reshape(original_shape)

    # STFT：segment → 时频复数矩阵
    # freqs shape: (nfft//2+1,)
    # t_frames shape: (n_frames,)
    # Zxx shape: (N, nfft//2+1, n_frames)，复数
    freqs, t_frames, Zxx = stft(
        segment,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=-1,
    )

    # 构建频段掩码（在目标频率范围内的系数置零）
    freq_mask = (freqs >= low_hz) & (freqs <= high_hz)  # shape: (nfft//2+1,)

    # 置零目标频段的所有时间帧
    Zxx_masked = Zxx.copy()
    Zxx_masked[:, freq_mask, :] = 0.0  # (N, masked_freqs, n_frames) = 0

    # ISTFT：重建时域信号
    _, reconstructed = istft(
        Zxx_masked,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        time_axis=-1,
        freq_axis=-2,
    )

    # ISTFT 输出长度可能与原片段略有不同，截断或补零对齐
    seg_len = t_end - t_start
    rec_len = reconstructed.shape[-1]
    if rec_len >= seg_len:
        reconstructed = reconstructed[:, :seg_len]
    else:
        pad_width = seg_len - rec_len
        reconstructed = np.pad(reconstructed, ((0, 0), (0, pad_width)), mode="edge")

    ablated_flat[:, t_start:t_end] = reconstructed.astype(np.float32)
    return ablated_flat.reshape(original_shape)


def mask_all_bands(eeg: np.ndarray, fs: float = FS) -> np.ndarray:
    """全段全频带置零（对照组：验证频率信息总体必要性）。"""
    eeg = np.asarray(eeg, dtype=np.float32)
    # 在频域将全部非直流分量置零 = 仅保留均值
    dc = eeg.mean(axis=-1, keepdims=True)
    return np.broadcast_to(dc, eeg.shape).copy()


# ===========================================================================
# NeuroBridge 推理相关
# ===========================================================================

def build_dataset(args: argparse.Namespace) -> EEGPreImageDataset:
    return EEGPreImageDataset(
        subject_ids=[args.subject],
        eeg_data_dir=str(args.eeg_data_dir),
        selected_channels=SELECTED_CHANNELS,
        time_window=[0, 250],
        image_feature_dir=str(args.image_feature_dir),
        text_feature_dir="",
        image_aug=True,
        aug_image_feature_dirs=[str(args.aug_feature_dir)],
        average=True,
        _random=False,
        eeg_transform=None,
        train=False,
        image_test_aug=True,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
    )


def load_models(
    checkpoint_path: Path,
    input_feature_dim: int,
    device: torch.device,
) -> tuple[EEGProject, ProjectorLinear, ProjectorLinear]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    projector_weight = checkpoint["eeg_projector_state_dict"]["linear.weight"]
    output_feature_dim, _ = projector_weight.shape

    model = EEGProject(
        feature_dim=input_feature_dim,
        eeg_sample_points=250,
        channels_num=len(SELECTED_CHANNELS),
    ).to(device)
    eeg_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)
    image_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])

    for net in (model, eeg_projector, image_projector):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return model, eeg_projector, image_projector


def encode_eeg(
    eeg: np.ndarray,
    model: EEGProject,
    projector: ProjectorLinear,
    device: torch.device,
    batch_size: int = 200,
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start:start + batch_size]).to(device)
            outputs.append(projector(model(batch)).cpu())
    return torch.cat(outputs, dim=0)


def retrieval_metrics(
    eeg_feats: torch.Tensor,
    img_feats: torch.Tensor,
    correct_cols: torch.Tensor,
) -> dict[str, float]:
    eeg_feats = F.normalize(eeg_feats, p=2, dim=1)
    img_feats = F.normalize(img_feats, p=2, dim=1)
    sims = eeg_feats @ img_feats.T
    ranking = sims.argsort(dim=1, descending=True)
    ranks = (ranking == correct_cols[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1":        float((ranks <= 1).float().mean()),
        "top5":        float((ranks <= 5).float().mean()),
        "mean_rank":   float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
        "raw_preds":   (ranks <= 1).tolist(),
    }


# ===========================================================================
# 实验主逻辑
# ===========================================================================


# ===========================================================================
# Phase 3: Single-Window Full-Frequency STFT Masking
# ===========================================================================

PHASE3_CONDITIONS = [
    {"name": "baseline",      "time_window": None},
    {"name": "full_freq_T0",  "time_window": "T0_0-50ms"},
    {"name": "full_freq_T1",  "time_window": "T1_50-150ms"},
    {"name": "full_freq_T2",  "time_window": "T2_150-300ms"},   # Core hypothesis: largest drop
    {"name": "full_freq_T3",  "time_window": "T3_300-500ms"},
    {"name": "full_freq_T4",  "time_window": "T4_500-800ms"},
]


def stft_mask_full_freq_window(
    eeg: np.ndarray,
    time_window: tuple,
    fs: float = FS,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    nfft: int = STFT_NFFT,
) -> np.ndarray:
    """
    Phase 3: zero ALL STFT frequency bins within the target time window.

    Equivalent to replacing that segment with a near-DC (flat) signal,
    erasing ALL oscillatory information in that window.

    vs Phase 1: Phase 1 masks one frequency band per run.
                Phase 3 masks every band simultaneously per run.
                -> answers "which TIME WINDOW carries the overall decoding information"

    The DC component (bin 0) is preserved to maintain the signal mean.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    original_shape = eeg.shape
    t_start, t_end = time_window

    flat    = eeg.reshape(-1, eeg.shape[-1])
    segment = flat[:, t_start:t_end]

    if segment.shape[-1] < nperseg:
        dc = segment.mean(axis=-1, keepdims=True)
        ablated = np.broadcast_to(dc, segment.shape).copy().astype(np.float32)
        result = flat.copy()
        result[:, t_start:t_end] = ablated
        return result.reshape(original_shape)

    freqs, _, Zxx = stft(
        segment, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1,
    )
    Zxx_masked = np.zeros_like(Zxx)
    Zxx_masked[:, 0, :] = Zxx[:, 0, :]   # preserve DC

    _, reconstructed = istft(
        Zxx_masked, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        time_axis=-1, freq_axis=-2,
    )
    seg_len = t_end - t_start
    rec_len = reconstructed.shape[-1]
    if rec_len >= seg_len:
        reconstructed = reconstructed[:, :seg_len]
    else:
        reconstructed = np.pad(reconstructed, ((0,0),(0, seg_len-rec_len)), mode="edge")

    result = flat.copy()
    result[:, t_start:t_end] = reconstructed.astype(np.float32)
    return result.reshape(original_shape)


def run_phase3(
    original_eeg: np.ndarray,
    model,
    eeg_projector,
    projected_images,
    correct_cols,
    device,
    baseline_metrics: dict,
    output_dir: Path,
) -> list:
    """Run Phase 3: full-frequency masking for each time window."""
    print("\n" + "="*60)
    print("[Phase 3] Full-Frequency Window Masking")
    print("  Design: zero ALL freq bins in ONE time window per run")
    print("  Hypothesis: full_freq_T2 (152-300ms) -> largest drop")
    print("="*60)

    results_p3 = []
    for cond in PHASE3_CONDITIONS:
        name   = cond["name"]
        tw_key = cond["time_window"]
        tw_ms  = ""
        if tw_key and tw_key in TIME_WINDOWS:
            tw = TIME_WINDOWS[tw_key]
            tw_ms = f" [{round(tw[0]/FS*1000)}-{round(tw[1]/FS*1000)}ms]"

        print(f"  {name}{tw_ms} ...", end=" ", flush=True)

        if tw_key is None:
            ablated = original_eeg
        else:
            tw = TIME_WINDOWS[tw_key]
            ablated = stft_mask_full_freq_window(original_eeg, tw)

        feats   = encode_eeg(ablated, model, eeg_projector, device)
        metrics = retrieval_metrics(feats, projected_images, correct_cols)

        result = {
            "name":        name,
            "time_window": tw_key,
            "time_ms":     tw_ms.strip("[] ") if tw_key else "—",
            "top1":        metrics["top1"],
            "top5":        metrics["top5"],
            "mean_rank":   metrics["mean_rank"],
            "median_rank": metrics["median_rank"],
            "top1_drop":   None if name == "baseline" else round(baseline_metrics["top1"] - metrics["top1"], 4),
            "top5_drop":   None if name == "baseline" else round(baseline_metrics["top5"] - metrics["top5"], 4),
            "raw_preds":   metrics["raw_preds"],
        }
        results_p3.append(result)

        drop_str = ""
        if result["top1_drop"] is not None:
            drop_str = f"  Delta_top1={result['top1_drop']:+.4f}  Delta_top5={result['top5_drop']:+.4f}"
        print(f"Top-1={metrics['top1']:.4f}  Top-5={metrics['top5']:.4f}{drop_str}")

    # Save CSV
    csv_path = output_dir / "phase3_full_freq_masking_results.csv"
    fieldnames = ["name","time_window","time_ms","top1","top5","mean_rank","median_rank","top1_drop","top5_drop"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_p3)
    print(f"\nSaved Phase 3 CSV  -> {csv_path}")

    # Save JSON
    json_path = output_dir / "phase3_full_freq_masking_details.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "experiment":   "Phase 3: Full-Frequency STFT Masking per Time Window",
            "description":  "Each run zeros ALL STFT bins in ONE time window, testing overall time-window importance",
            "hypothesis":   "full_freq_T2 (152-300ms) should produce the largest accuracy drop",
            "reference":    "Thorpe et al. (1996) Nature 381:520; Cichy et al. (2014) Nat Neurosci 17:455",
            "time_windows": {k: {"samples": list(v), "ms": [round(v[0]/FS*1000), round(v[1]/FS*1000)]}
                             for k, v in TIME_WINDOWS.items() if k != "T_full"},
            "freq_bands_masked": "ALL (DC preserved to maintain signal mean)",
            "baseline":     baseline_metrics,
            "results":      results_p3,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved Phase 3 JSON -> {json_path}")

    # Ranking summary
    ranked = sorted([r for r in results_p3 if r["top1_drop"] is not None],
                    key=lambda x: x["top1_drop"], reverse=True)
    print("\n[Phase 3] Time Window Importance Ranking:")
    print(f"  {'Condition':<22} {'Time':>10} {'DeltaTop-1':>12} {'DeltaTop-5':>12}")
    print("  " + "-"*58)
    for r in ranked:
        bar = chr(9608) * max(0, int(r["top1_drop"] * 80))
        print(f"  {r['name']:<22} {r['time_ms']:>10} {r['top1_drop']:>+12.4f} {r['top5_drop']:>+12.4f}  {bar}")

    return results_p3


def build_experiment_list(mode: str) -> list[dict]:
    """
    根据模式生成实验列表。

    mode:
      "priority"  → 只跑优先级 ≥4 的组合（必做6个 + 对照组）
      "standard"  → 跑优先级 ≥3 的组合（核心12个 + 对照组）
      "full"      → 跑所有组合（30个 + 对照组）
    """
    min_priority = {"priority": 4, "standard": 3, "full": 1}[mode]

    experiments = []
    for (tw_name, fb_name), priority in PRIORITY_MATRIX.items():
        if priority >= min_priority:
            experiments.append({
                "name":        f"{tw_name}__{fb_name}",
                "time_window": tw_name,
                "freq_band":   fb_name,
                "priority":    priority,
            })

    # 按优先级从高到低排序
    experiments.sort(key=lambda x: -x["priority"])

    # 添加对照组（始终运行）
    for ctrl in CONTROL_CONDITIONS:
        experiments.insert(0 if ctrl["name"] == "baseline" else len(experiments), ctrl)

    return experiments


def run_experiment(
    exp: dict,
    original_eeg: np.ndarray,
    model: EEGProject,
    eeg_projector: ProjectorLinear,
    projected_images: torch.Tensor,
    correct_cols: torch.Tensor,
    device: torch.device,
    baseline_metrics: dict | None = None,
) -> dict:
    """运行单个消融实验，返回结果字典。"""

    name = exp["name"]
    tw_name = exp.get("time_window")
    fb_name = exp.get("freq_band")

    # 生成消融 EEG
    if tw_name is None and fb_name is None:
        # Baseline：不做任何处理
        ablated_eeg = original_eeg
    elif fb_name == "all_bands":
        # 全频带置零对照
        ablated_eeg = mask_all_bands(original_eeg)
    else:
        tw = TIME_WINDOWS[tw_name]
        fb = FREQ_BANDS[fb_name]
        ablated_eeg = stft_mask_time_freq(original_eeg, tw, fb)

    # 推理
    eeg_feats = encode_eeg(ablated_eeg, model, eeg_projector, device)
    metrics = retrieval_metrics(eeg_feats, projected_images, correct_cols)

    result = {
        "name":        name,
        "time_window": tw_name,
        "freq_band":   fb_name,
        "priority":    exp.get("priority", "control"),
        **metrics,
    }

    # 计算相对于 baseline 的下降
    if baseline_metrics is not None and name != "baseline":
        result["top1_drop"] = round(baseline_metrics["top1"] - metrics["top1"], 4)
        result["top5_drop"] = round(baseline_metrics["top5"] - metrics["top5"], 4)
    else:
        result["top1_drop"] = None
        result["top5_drop"] = None

    return result


# ===========================================================================
# 入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: STFT Time-Frequency Masking Ablation for NeuroBridge"
    )
    parser.add_argument("--subject",    type=int,  default=8)
    parser.add_argument("--device",     type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int,  default=200)
    parser.add_argument(
        "--mode",
        choices=["priority", "standard", "full"],
        default="priority",
        help=(
            "priority: 只跑优先级≥4（6+对照，最快）; "
            "standard: 优先级≥3（12+对照）; "
            "full: 全部组合（30+对照）"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "intra-subjects_sub-08_checkpoint_last.pth",
    )
    parser.add_argument(
        "--eeg-data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "things_eeg" / "preprocessed_eeg",
    )
    parser.add_argument(
        "--image-feature-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "things_eeg" / "image_feature" / "RN50",
    )
    parser.add_argument(
        "--aug-feature-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "things_eeg" / "image_feature" / "RN50" / "GaussianBlur-GaussianNoise-LowResolution-Mosaic",
        help="aug image feature dir (same as evaluate.py aug_feature_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "temporal_stft_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Phase 1] STFT Temporal Ablation | subject={args.subject} | device={device} | mode={args.mode}")

    # 加载数据
    print("Loading test EEG data...")
    dataset = build_dataset(args)
    loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch   = next(iter(loader))

    original_eeg    = batch[0].numpy().astype(np.float32)   # (N, 17, 250)
    raw_img_feats = batch[1].to(device)   # aug图特征，与evaluate.py/论文一致
    object_indices  = batch[4].numpy()
    image_indices   = batch[5].numpy()
    print(f"  EEG shape: {original_eeg.shape}  (N, channels, time_points)")
    print(f"  FS={FS}Hz → 1 sample = {1000/FS:.1f}ms, total = {original_eeg.shape[-1]/FS*1000:.0f}ms")

    # 加载模型
    print("Loading NeuroBridge model (frozen)...")
    model, eeg_projector, image_projector = load_models(
        args.checkpoint, raw_img_feats.shape[-1], device
    )
    with torch.inference_mode():
        projected_images = image_projector(raw_img_feats).cpu()

    # correct_cols: 原始image_test.npy按object顺序排列，EEG也按object顺序 → 直接arange
    correct_cols   = torch.arange(len(raw_img_feats), dtype=torch.long)

    # 生成实验列表
    experiments = build_experiment_list(args.mode)
    print(f"\nExperiments to run: {len(experiments)}")
    for exp in experiments:
        prio = exp.get('priority', 'ctrl')
        print(f"  {'★' * (prio if isinstance(prio, int) else 1):<6} {exp['name']}")

    # 运行实验
    results = []
    baseline_metrics = None

    print("\n" + "="*60)
    for i, exp in enumerate(experiments):
        print(f"[{i+1:02d}/{len(experiments):02d}] {exp['name']} ...", end=" ", flush=True)
        result = run_experiment(
            exp, original_eeg, model, eeg_projector,
            projected_images, correct_cols, device, baseline_metrics
        )
        if exp["name"] == "baseline":
            baseline_metrics = result
        results.append(result)

        drop_str = ""
        if result["top1_drop"] is not None:
            drop_str = f"  Δtop1={result['top1_drop']:+.4f}  Δtop5={result['top5_drop']:+.4f}"
        print(f"Top-1={result['top1']:.4f}  Top-5={result['top5']:.4f}{drop_str}")

    # 保存结果
    # CSV
    csv_path = args.output_dir / "stft_ablation_results.csv"
    fieldnames = ["name", "time_window", "freq_band", "priority",
                  "top1", "top5", "mean_rank", "median_rank", "top1_drop", "top5_drop"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV  → {csv_path}")

    # JSON（完整配置+结果）
    json_path = args.output_dir / "stft_ablation_details.json"
    payload = {
        "experiment": "Phase 1: STFT Time-Frequency Masking Ablation",
        "subject":    args.subject,
        "mode":       args.mode,
        "fs_hz":      FS,
        "stft_params": {
            "nperseg":  STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "nfft":     STFT_NFFT,
            "window":   "hann",
            "time_resolution_ms": round(STFT_NPERSEG / FS * 1000, 1),
            "freq_resolution_hz": round(FS / STFT_NFFT, 2),
        },
        "time_windows": {k: {"samples": list(v), "ms": [round(v[0]/FS*1000), round(v[1]/FS*1000)]}
                         for k, v in TIME_WINDOWS.items()},
        "freq_bands":   {k: list(v) for k, v in FREQ_BANDS.items()},
        "eeg_shape":    list(original_eeg.shape),
        "checkpoint":   str(args.checkpoint),
        "baseline":     baseline_metrics,
        "results":      results,
        "references": {
            "time_windows": "Cichy et al. (2014) Nat Neurosci 17(3):455-462; Thorpe et al. (1996) Nature 381:520-522",
            "freq_bands":   "Fries (2015) Neuron 88:220-235; Wang et al. (2026) IEEE JBHI 30(4):3543",
            "dataset":      "Gifford et al. (2022) NeuroImage 264:119754",
            "model":        "Zhang et al. (2025) NeuroBridge",
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {json_path}")

    # 打印排序后的 Top-1 Drop 摘要
    ranked = sorted(
        [r for r in results if r["top1_drop"] is not None],
        key=lambda x: x["top1_drop"], reverse=True
    )
    print("\n" + "="*60)
    print("[CHART] Top-1 Accuracy Drop Ranking (高 = 该组合对模型更重要):")
    print(f"  {'Condition':<35} {'Top-1 Drop':>10} {'Top-5 Drop':>10}")
    print("  " + "-"*57)
    for r in ranked:
        bar = "█" * max(0, int(r["top1_drop"] * 100))
        print(f"  {r['name']:<35} {r['top1_drop']:>+10.4f} {r['top5_drop']:>+10.4f}  {bar}")
    print("="*60)
    # =============================================================
    # Phase 3: Full-Frequency Window Masking (auto-run after Phase 1)
    # =============================================================
    print("\n[INFO] Auto-running Phase 3 (full-freq window masking)...")
    p3_output = args.output_dir.parent / "phase3_full_freq_masking"
    p3_output.mkdir(parents=True, exist_ok=True)
    run_phase3(original_eeg, model, eeg_projector, projected_images,
               correct_cols, device, baseline_metrics, p3_output)

    print("\n[OK] Phase 1 STFT Ablation complete.")
    print(f"   Results: {args.output_dir}")
    print(f"   Next step: Run ablation_temporal_amplitude.py --top-conditions <name1> <name2> <name3>")


if __name__ == "__main__":
    main()






