"""
ablation_temporal_amplitude.py — Phase 2: Amplitude Perturbation Ablation
==========================================================================
针对 NeuroBridge EEG-to-Image 解码模型的第二阶段 Temporal 振幅扰动实验。

使用方式：
  先运行 ablation_temporal_stft.py（Phase 1），找到 top-N 关键时频组合，
  再运行本脚本对这些组合做三类振幅扰动：
    A. 振幅缩放（Amplitude Scaling）    — 测试模型对绝对功率的依赖
    B. 相位随机化（Phase Randomization）— 分离相位 vs 振幅的贡献
    C. 高斯噪声注入（Gaussian Noise）   — 测试信噪比鲁棒性

实验设计依据：
  - Haufe et al. (2014) NeuroImage 87:96-110：前向模型解释框架
  - Fries (2015) Neuron 88:220-235：gamma/beta 功能分工
  - AD/SCD 关联：振幅缩放 α≈0.6-0.8 模拟 AD 早期振幅退化

用法示例：
  # 自动读取 Phase 1 结果，取 top-3 组合
  python ablation_temporal_amplitude.py

  # 手动指定消融目标组合
  python ablation_temporal_amplitude.py \\
      --conditions T2_150-300ms__gamma T1_50-150ms__gamma T2_150-300ms__beta

  # 只跑振幅缩放
  python ablation_temporal_amplitude.py --types scaling

作者：乔钰成 / NEOschool 项目组
版本：1.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING


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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import istft, stft
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from module.dataset import EEGPreImageDataset
    from module.eeg_encoder.model import EEGProject
    from module.projector import ProjectorLinear

# ===========================================================================
# 常量配置（与 Phase 1 保持一致）
# ===========================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_DATA  = REPO_ROOT / "data"
FS           = 250.0

SELECTED_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

# STFT 参数（与 Phase 1 一致）
STFT_NPERSEG = 32
STFT_NOVERLAP = 16
STFT_NFFT    = 64

TIME_WINDOWS = {
    "T0_0-50ms":    (0,   13),
    "T1_50-150ms":  (13,  38),
    "T2_150-300ms": (38,  75),
    "T3_300-500ms": (75,  125),
    "T4_500-800ms": (125, 200),
    "T_full":       (0,   250),
}

FREQ_BANDS = {
    "delta":    (1.0,  4.0),
    "theta":    (4.0,  8.0),
    "alpha":    (8.0,  13.0),
    "beta":     (13.0, 30.0),
    "gamma":    (30.0, 80.0),
    "hi_gamma": (80.0, 120.0),
}

# ---------------------------------------------------------------------------
# Phase 2 扰动参数
# ---------------------------------------------------------------------------

# A. 振幅缩放系数
# α=0.0 等价于 Phase 1 置零（桥梁实验）；α=1.0 为 baseline；α>1.0 为增强
# 依据：AD 振幅退化约 20-40%（α≈0.6-0.8），需覆盖该范围
ALPHA_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

# B. 相位随机化程度（0=不随机化，1=完全随机化）
PHASE_RAND_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.00]

# C. 高斯噪声 SNR（dB）
# SNR(dB) = 10 * log10(P_signal / P_noise)
# 正值=信号强于噪声，0=等功率，负值=噪声强于信号
SNR_LEVELS_DB = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0]

# 默认 top-N 条件（若未手动指定，从 Phase 1 结果读取）
DEFAULT_TOP_N = 3
DEFAULT_PHASE1_CSV = SCRIPT_DIR / "results" / "temporal_stft_ablation" / "stft_ablation_results.csv"


# ===========================================================================
# 振幅扰动核心函数
# ===========================================================================

def _build_time_freq_representation(
    eeg: np.ndarray,
    t_start: int,
    t_end: int,
    low_hz: float,
    high_hz: float,
) -> dict[str, np.ndarray | tuple | str | int]:
    """
    内部函数：获取目标时间窗片段的频域表示及频段掩码。

    当时间窗长度 < STFT_NPERSEG 时，退化为 FFT 路径，保持与 Phase 1 一致。
    """
    flat = eeg.reshape(-1, eeg.shape[-1])
    segment = flat[:, t_start:t_end]
    seg_len = segment.shape[-1]

    if seg_len < STFT_NPERSEG:
        coeffs = np.fft.rfft(segment, axis=-1)
        freqs = np.fft.rfftfreq(seg_len, d=1.0 / FS)
        freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
        return {
            "mode": "fft",
            "flat": flat,
            "coeffs": coeffs,
            "freq_mask": freq_mask,
            "seg_len": seg_len,
            "original_shape": eeg.shape,
            "t_start": t_start,
            "t_end": t_end,
        }

    freqs, _, coeffs = stft(
        segment,
        fs=FS,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        axis=-1,
    )
    freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
    return {
        "mode": "stft",
        "flat": flat,
        "coeffs": coeffs,
        "freq_mask": freq_mask,
        "seg_len": seg_len,
        "original_shape": eeg.shape,
        "t_start": t_start,
        "t_end": t_end,
    }


def _invert_time_freq_representation(
    rep: dict[str, np.ndarray | tuple | str | int],
    coeffs_modified: np.ndarray,
) -> np.ndarray:
    """内部函数：频域重建并填回原始 EEG。"""
    flat = rep["flat"]
    t_start = int(rep["t_start"])
    t_end = int(rep["t_end"])
    seg_len = int(rep["seg_len"])
    original_shape = rep["original_shape"]
    mode = rep["mode"]

    if mode == "fft":
        reconstructed = np.fft.irfft(coeffs_modified, n=seg_len, axis=-1)
    else:
        _, reconstructed = istft(
            coeffs_modified,
            fs=FS,
            window="hann",
            nperseg=STFT_NPERSEG,
            noverlap=STFT_NOVERLAP,
            nfft=STFT_NFFT,
            time_axis=-1,
            freq_axis=-2,
        )

    rec_len = reconstructed.shape[-1]
    if rec_len >= seg_len:
        reconstructed = reconstructed[:, :seg_len]
    else:
        reconstructed = np.pad(reconstructed, ((0, 0), (0, seg_len - rec_len)), mode="edge")

    result = flat.copy()
    result[:, t_start:t_end] = reconstructed.astype(np.float32)
    return result.reshape(original_shape)


def _randomize_phase_values(
    target: np.ndarray,
    rand_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按 rand_ratio 抽样替换相位，振幅保持不变。"""
    amplitude = np.abs(target)
    original_phase = np.angle(target)
    if rand_ratio <= 0.0:
        return amplitude * np.exp(1j * original_phase)
    if rand_ratio >= 1.0:
        selection_mask = np.ones(target.shape, dtype=bool)
    else:
        selection_mask = rng.random(target.shape) < rand_ratio
    random_phase = rng.uniform(0.0, 2 * np.pi, size=target.shape)
    new_phase = np.where(selection_mask, random_phase, original_phase)
    return amplitude * np.exp(1j * new_phase)


# --- A. 振幅缩放 ---

def amplitude_scaling(
    eeg: np.ndarray,
    time_window: tuple[int, int],
    freq_band: tuple[float, float],
    alpha: float,
) -> np.ndarray:
    """
    在目标时频区域，将 STFT 系数振幅乘以 alpha，保留相位不变。

    alpha=0.0 → 等价于 Phase 1 置零
    alpha=1.0 → 无变化（baseline）
    alpha>1.0 → 振幅增强

    数学公式：
      Zxx_scaled[f, t] = |Zxx[f, t]| * alpha * exp(j * angle(Zxx[f, t]))
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    t_start, t_end = time_window
    low_hz, high_hz = freq_band

    rep = _build_time_freq_representation(eeg, t_start, t_end, low_hz, high_hz)
    coeffs = rep["coeffs"]
    freq_mask = rep["freq_mask"]
    coeffs_mod = coeffs.copy()
    # 仅对目标频带缩放振幅，相位保留
    if rep["mode"] == "fft":
        coeffs_mod[:, freq_mask] = coeffs[:, freq_mask] * alpha
    else:
        coeffs_mod[:, freq_mask, :] = coeffs[:, freq_mask, :] * alpha
    return _invert_time_freq_representation(rep, coeffs_mod)


# --- B. 相位随机化 ---

def phase_randomization(
    eeg: np.ndarray,
    time_window: tuple[int, int],
    freq_band: tuple[float, float],
    rand_ratio: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    在目标时频区域，将 rand_ratio 比例的 STFT 系数相位随机化，振幅不变。

    rand_ratio=0.0 → 不随机化（baseline）
    rand_ratio=1.0 → 完全随机相位（保留功率谱，破坏时序结构）

    数学公式（完全随机化情形）：
      Zxx_phase_rand[f, t] = |Zxx[f, t]| * exp(j * U(0, 2π))

    与振幅缩放对比：
      相位随机化后准确率大幅下降 → 模型依赖精确相位/时序结构
      相位随机化后准确率不变    → 模型依赖功率谱，不依赖相位
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    t_start, t_end = time_window
    low_hz, high_hz = freq_band
    if rng is None:
        rng = np.random.default_rng(seed=42)

    rep = _build_time_freq_representation(eeg, t_start, t_end, low_hz, high_hz)
    coeffs = rep["coeffs"]
    freq_mask = rep["freq_mask"]
    coeffs_mod = coeffs.copy()

    if rep["mode"] == "fft":
        target = coeffs[:, freq_mask]
        coeffs_mod[:, freq_mask] = _randomize_phase_values(target, rand_ratio, rng)
    else:
        target = coeffs[:, freq_mask, :]
        coeffs_mod[:, freq_mask, :] = _randomize_phase_values(target, rand_ratio, rng)
    return _invert_time_freq_representation(rep, coeffs_mod)


# --- C. 高斯噪声注入 ---

def gaussian_noise_injection(
    eeg: np.ndarray,
    time_window: tuple[int, int],
    freq_band: tuple[float, float],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    在目标时频区域，向 STFT 系数（实部+虚部）注入高斯噪声，达到指定 SNR。

    SNR(dB) = 10 * log10(P_signal / P_noise)
    SNR=20dB → 轻微干扰（P_noise = P_signal / 100）
    SNR=0dB  → 等功率干扰
    SNR=-10dB → 强干扰（P_noise = 10 * P_signal）

    与振幅缩放的本质区别：
      噪声注入保留了原始信号但叠加了干扰 → 测量「信噪比容忍度」
      振幅缩放保留了信号结构但改变了幅度  → 测量「绝对功率依赖」
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    t_start, t_end = time_window
    low_hz, high_hz = freq_band
    if rng is None:
        rng = np.random.default_rng(seed=42)

    rep = _build_time_freq_representation(eeg, t_start, t_end, low_hz, high_hz)
    coeffs = rep["coeffs"]
    freq_mask = rep["freq_mask"]
    coeffs_mod = coeffs.copy()
    if rep["mode"] == "fft":
        target = coeffs[:, freq_mask]
    else:
        target = coeffs[:, freq_mask, :]

    # 计算目标区域的信号功率（实部和虚部分别计算）
    signal_power = (np.abs(target) ** 2).mean()
    if signal_power < 1e-12:
        return eeg.copy()  # 信号本身接近零，无意义

    # 根据 SNR 计算噪声标准差
    # SNR(linear) = 10^(SNR_dB / 10)
    snr_linear  = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std   = np.sqrt(noise_power / 2)  # 除以2因为复数有实部和虚部

    # 向复数 STFT 系数注入复高斯噪声
    noise = rng.normal(0, noise_std, size=target.shape) + \
            1j * rng.normal(0, noise_std, size=target.shape)
    if rep["mode"] == "fft":
        coeffs_mod[:, freq_mask] = target + noise.astype(np.complex64)
    else:
        coeffs_mod[:, freq_mask, :] = target + noise.astype(np.complex64)
    return _invert_time_freq_representation(rep, coeffs_mod)


# ===========================================================================
# NeuroBridge 推理（与 Phase 1 完全相同）
# ===========================================================================

def build_dataset(args: argparse.Namespace) -> EEGPreImageDataset:
    from module.dataset import EEGPreImageDataset
    selected_channels = getattr(args, "selected_channels", SELECTED_CHANNELS)

    return EEGPreImageDataset(
        subject_ids=[args.subject],
        eeg_data_dir=str(args.eeg_data_dir),
        selected_channels=selected_channels,
        time_window=[0, 250],
        image_feature_dir=str(args.image_feature_dir),
        text_feature_dir="",
        image_aug=True, aug_image_feature_dirs=[str(args.aug_feature_dir)],
        average=True, _random=False, eeg_transform=None,
        train=False, image_test_aug=True, eeg_test_aug=False, frozen_eeg_prior=False,
    )


def load_models(ckpt: Path, feat_dim: int, device: torch.device):
    from module.eeg_encoder.model import EEGProject, TSConv
    from module.projector import ProjectorLinear

    ckpt_data = torch.load(ckpt, map_location=device)
    out_dim, _ = ckpt_data["eeg_projector_state_dict"]["linear.weight"].shape
    model_state_dict = ckpt_data["model_state_dict"]
    uses_tsconv = any(key.startswith("tsconv.") or key.startswith("proj_eeg.") for key in model_state_dict)
    model_cls = TSConv if uses_tsconv else EEGProject
    if uses_tsconv:
        channels_num = int(model_state_dict["tsconv.4.weight"].shape[2])
    else:
        channels_num = int(model_state_dict["model.0.weight"].shape[1] // 250)
    model = model_cls(feature_dim=feat_dim, eeg_sample_points=250,
                      channels_num=channels_num).to(device)
    eeg_proj = ProjectorLinear(feat_dim, out_dim).to(device)
    img_proj = ProjectorLinear(feat_dim, out_dim).to(device)
    model.load_state_dict(model_state_dict)
    eeg_proj.load_state_dict(ckpt_data["eeg_projector_state_dict"])
    img_proj.load_state_dict(ckpt_data["img_projector_state_dict"])
    for net in (model, eeg_proj, img_proj):
        net.eval()
        for p in net.parameters(): p.requires_grad_(False)
    return model, eeg_proj, img_proj


def encode_eeg(eeg, model, proj, device, batch_size=200):
    out = []
    with torch.inference_mode():
        for s in range(0, len(eeg), batch_size):
            b = torch.from_numpy(eeg[s:s+batch_size]).to(device)
            out.append(proj(model(b)).cpu())
    return torch.cat(out)


def retrieval_metrics(eeg_f, img_f, correct_cols):
    eeg_f = F.normalize(eeg_f, p=2, dim=1)
    img_f = F.normalize(img_f, p=2, dim=1)
    sims  = eeg_f @ img_f.T
    ranks = (sims.argsort(dim=1, descending=True) == correct_cols[:,None]).nonzero(as_tuple=False)[:,1] + 1
    return {
        "top1":        float((ranks<=1).float().mean()),
        "top5":        float((ranks<=5).float().mean()),
        "mean_rank":   float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
        "raw_preds":   (ranks <= 1).tolist(),
    }


# ===========================================================================
# 实验条件解析
# ===========================================================================

def load_top_conditions_from_phase1(csv_path: Path, top_n: int) -> list[str]:
    """从 Phase 1 CSV 结果中读取 top-N 条件名（按 top1_drop 降序）。"""
    if not csv_path.exists():
        print(f"[Warning] Phase 1 CSV not found: {csv_path}")
        print("  Using default top conditions: T2_150-300ms__gamma, T1_50-150ms__gamma, T2_150-300ms__beta")
        return ["T2_150-300ms__gamma", "T1_50-150ms__gamma", "T2_150-300ms__beta"]

    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["name"] not in ("baseline", "full_mask_all", "random_control") \
               and row["top1_drop"] not in ("", "None", None):
                try:
                    rows.append((row["name"], float(row["top1_drop"])))
                except ValueError:
                    pass
    rows.sort(key=lambda x: -x[1])
    top = [r[0] for r in rows[:top_n]]
    print(f"[Phase 1 → Phase 2] Top-{top_n} conditions by Top-1 Drop: {top}")
    return top


def parse_condition(condition_name: str) -> tuple[str, str]:
    """将条件名 'T2_150-300ms__gamma' 解析为 (time_window_key, freq_band_key)。"""
    parts = condition_name.split("__")
    if len(parts) != 2:
        raise ValueError(f"Invalid condition name: {condition_name}. Expected format: '<time_window>__<freq_band>'")
    return parts[0], parts[1]


# ===========================================================================
# 主逻辑
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Amplitude Perturbation Ablation for NeuroBridge"
    )
    parser.add_argument("--subject",     type=int,  default=8)
    parser.add_argument("--device",      type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size",  type=int,  default=200)
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="手动指定目标条件（如 T2_150-300ms__gamma T1_50-150ms__gamma）。"
             "若不指定，自动从 Phase 1 CSV 读取 top-N。"
    )
    parser.add_argument("--top-n",       type=int,  default=DEFAULT_TOP_N,
                        help="从 Phase 1 自动读取的 top-N 条件数量（默认3）。")
    parser.add_argument(
        "--types", nargs="+",
        choices=["scaling", "phase", "noise"],
        default=["scaling", "phase", "noise"],
        help="要运行的扰动类型（默认全部）。"
    )
    parser.add_argument(
        "--phase1-csv", type=Path, default=DEFAULT_PHASE1_CSV,
        help="Phase 1 结果 CSV 路径（用于自动读取 top 条件）。"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=REPO_ROOT / "checkpoints" / "intra-subjects_sub-08_checkpoint_last.pth",
        help="Path to the pretrained checkpoint (place it under <repo>/checkpoints/ or override via CLI).",
    )
    parser.add_argument(
        "--eeg-data-dir", type=Path,
        default=DEFAULT_DATA / "things_eeg" / "preprocessed_eeg",
    )
    parser.add_argument(
        "--image-feature-dir", type=Path,
        default=DEFAULT_DATA / "things_eeg" / "image_feature" / "RN50",
    )
    parser.add_argument(
        "--aug-feature-dir",
        type=Path,
        default=DEFAULT_DATA / "things_eeg" / "image_feature" / "RN50" / "GaussianBlur-GaussianNoise-LowResolution-Mosaic",
        help="aug image feature dir (same as evaluate.py aug_feature_dir)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=SCRIPT_DIR / "results" / "temporal_amplitude_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng    = np.random.default_rng(seed=42)

    print(f"[Phase 2] Amplitude Perturbation | subject={args.subject} | device={device}")
    print(f"  Perturbation types: {args.types}")

    # 确定目标条件
    if args.conditions:
        conditions = args.conditions
        print(f"  Conditions (manual): {conditions}")
    else:
        conditions = load_top_conditions_from_phase1(args.phase1_csv, args.top_n)

    # 加载数据
    print("Loading test EEG data...")
    dataset = build_dataset(args)
    loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch   = next(iter(loader))
    original_eeg   = batch[0].numpy().astype(np.float32)
    raw_img_feats  = batch[1].to(device)
    object_indices = batch[4].numpy()
    image_indices  = batch[5].numpy()
    print(f"  EEG shape: {original_eeg.shape}")

    # 加载模型
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Place the pretrained checkpoint at <repo>/checkpoints/ "
            "(see README) or pass --checkpoint <path>."
        )
    print("Loading NeuroBridge model (frozen)...")
    model, eeg_proj, img_proj = load_models(args.checkpoint, raw_img_feats.shape[-1], device)
    with torch.inference_mode():
        projected_images = img_proj(raw_img_feats).cpu()

    gallery_keys = list(zip(object_indices.tolist(), image_indices.tolist()))
    key_to_col   = {k: i for i, k in enumerate(gallery_keys)}
    correct_cols = torch.tensor([key_to_col[k] for k in gallery_keys], dtype=torch.long)

    # Baseline
    print("\nRunning baseline...")
    baseline_feats   = encode_eeg(original_eeg, model, eeg_proj, device)
    baseline_metrics = retrieval_metrics(baseline_feats, projected_images, correct_cols)
    print(f"  Baseline Top-1={baseline_metrics['top1']:.4f}, Top-5={baseline_metrics['top5']:.4f}")

    # 生成所有实验
    all_results = []

    for cond_name in conditions:
        tw_key, fb_key = parse_condition(cond_name)
        tw = TIME_WINDOWS[tw_key]
        fb = FREQ_BANDS[fb_key]
        tw_ms = f"{round(tw[0]/FS*1000)}-{round(tw[1]/FS*1000)}ms"
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}  [{tw_ms}, {fb_key} {fb[0]}-{fb[1]}Hz]")

        # --- A. 振幅缩放 ---
        if "scaling" in args.types:
            print("  [A] Amplitude Scaling:")
            for alpha in ALPHA_LEVELS:
                ablated = amplitude_scaling(original_eeg, tw, fb, alpha)
                feats   = encode_eeg(ablated, model, eeg_proj, device)
                m       = retrieval_metrics(feats, projected_images, correct_cols)
                result  = {
                    "condition":    cond_name,
                    "perturbation": "scaling",
                    "param_name":   "alpha",
                    "param_value":  alpha,
                    "top1":         m["top1"],
                    "top5":         m["top5"],
                    "mean_rank":    m["mean_rank"],
                    "median_rank":  m["median_rank"],
                    "top1_drop":    round(baseline_metrics["top1"] - m["top1"], 4),
                    "top5_drop":    round(baseline_metrics["top5"] - m["top5"], 4),
                    "raw_preds":    m["raw_preds"],
                }
                all_results.append(result)
                marker = " ← AD range" if 0.5 <= alpha <= 0.85 else ""
                print(f"    α={alpha:.2f}  Top-1={m['top1']:.4f}  Δ={result['top1_drop']:+.4f}{marker}")

        # --- B. 相位随机化 ---
        if "phase" in args.types:
            print("  [B] Phase Randomization:")
            for rand_ratio in PHASE_RAND_LEVELS:
                ablated = phase_randomization(original_eeg, tw, fb, rand_ratio, rng)
                feats   = encode_eeg(ablated, model, eeg_proj, device)
                m       = retrieval_metrics(feats, projected_images, correct_cols)
                result  = {
                    "condition":    cond_name,
                    "perturbation": "phase_rand",
                    "param_name":   "rand_ratio",
                    "param_value":  rand_ratio,
                    "top1":         m["top1"],
                    "top5":         m["top5"],
                    "mean_rank":    m["mean_rank"],
                    "median_rank":  m["median_rank"],
                    "top1_drop":    round(baseline_metrics["top1"] - m["top1"], 4),
                    "top5_drop":    round(baseline_metrics["top5"] - m["top5"], 4),
                    "raw_preds":    m["raw_preds"],
                }
                all_results.append(result)
                print(f"    rand={rand_ratio:.2f}  Top-1={m['top1']:.4f}  Δ={result['top1_drop']:+.4f}")

        # --- C. 高斯噪声注入 ---
        if "noise" in args.types:
            print("  [C] Gaussian Noise Injection:")
            for snr_db in SNR_LEVELS_DB:
                ablated = gaussian_noise_injection(original_eeg, tw, fb, snr_db, rng)
                feats   = encode_eeg(ablated, model, eeg_proj, device)
                m       = retrieval_metrics(feats, projected_images, correct_cols)
                result  = {
                    "condition":    cond_name,
                    "perturbation": "gaussian_noise",
                    "param_name":   "snr_db",
                    "param_value":  snr_db,
                    "top1":         m["top1"],
                    "top5":         m["top5"],
                    "mean_rank":    m["mean_rank"],
                    "median_rank":  m["median_rank"],
                    "top1_drop":    round(baseline_metrics["top1"] - m["top1"], 4),
                    "top5_drop":    round(baseline_metrics["top5"] - m["top5"], 4),
                    "raw_preds":    m["raw_preds"],
                }
                all_results.append(result)
                print(f"    SNR={snr_db:+.0f}dB  Top-1={m['top1']:.4f}  Δ={result['top1_drop']:+.4f}")

    # 保存结果
    csv_path = args.output_dir / "amplitude_ablation_results.csv"
    fieldnames = ["condition", "perturbation", "param_name", "param_value",
                  "top1", "top5", "mean_rank", "median_rank", "top1_drop", "top5_drop"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved CSV  → {csv_path}")

    json_path = args.output_dir / "amplitude_ablation_details.json"
    payload = {
        "experiment":    "Phase 2: Amplitude Perturbation Ablation",
        "subject":       args.subject,
        "conditions":    conditions,
        "perturbation_types": args.types,
        "params": {
            "alpha_levels":      ALPHA_LEVELS,
            "phase_rand_levels": PHASE_RAND_LEVELS,
            "snr_levels_db":     SNR_LEVELS_DB,
        },
        "baseline":  baseline_metrics,
        "results":   all_results,
        "interpretation_guide": {
            "scaling_no_drop":      "α大幅降低但准确率不变 → 模型对绝对振幅不敏感，依赖频率模式",
            "scaling_linear_drop":  "准确率随α线性下降 → 功率本身是特征",
            "scaling_threshold":    "找到准确率下降超过5%的临界α → 模型的振幅容忍下限",
            "ad_range_alpha":       "α=0.6-0.8 模拟AD振幅退化20-40%，观察临床意义",
            "phase_rand_big_drop":  "相位随机化后大幅下降 → 模型依赖精确相位/时序",
            "phase_vs_scaling":     "比较相位随机化 vs 振幅置零(α=0) 的drop → 分离相位/振幅贡献",
            "snr_threshold":        "找到准确率崩溃的临界SNR → BCI实际部署的噪声容忍下限",
        },
        "references": {
            "amplitude_framework": "Haufe et al. (2014) NeuroImage 87:96-110",
            "freq_function":       "Fries (2015) Neuron 88:220-235",
            "ad_amplitude":        "AD/SCD EEG amplitude reduction: spectral slowing evidence",
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {json_path}")

    # 关键发现摘要
    print("\n" + "="*60)
    print("📊 Phase 2 Key Findings Summary:")
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        print(f"\n  [{cond}]")
        # 振幅缩放：找临界 alpha
        scaling = [r for r in cond_results if r["perturbation"] == "scaling"]
        if scaling:
            threshold = next((r for r in scaling if r["top1_drop"] > 0.05), None)
            if threshold:
                print(f"    Scaling threshold (Δ>5%): α={threshold['param_value']:.2f}  "
                      f"(drop={threshold['top1_drop']:+.4f})")
        # 相位随机化 vs alpha=0
        phase_full = next((r for r in cond_results
                          if r["perturbation"]=="phase_rand" and r["param_value"]==1.0), None)
        scale_zero = next((r for r in cond_results
                          if r["perturbation"]=="scaling" and r["param_value"]==0.0), None)
        if phase_full and scale_zero:
            print(f"    Phase(100% rand) drop={phase_full['top1_drop']:+.4f}  "
                  f"vs  Scale(α=0) drop={scale_zero['top1_drop']:+.4f}")
            if phase_full["top1_drop"] < scale_zero["top1_drop"] - 0.02:
                print(f"    → 振幅信息比相位信息更关键（Power > Phase）")
            elif scale_zero["top1_drop"] < phase_full["top1_drop"] - 0.02:
                print(f"    → 相位信息比振幅信息更关键（Phase > Power）")
            else:
                print(f"    → 相位和振幅贡献相当")

    print("="*60)
    print("\n✅ Phase 2 Amplitude Perturbation Ablation complete.")
    print(f"   Results: {args.output_dir}")
    print(f"   Next step: Run ablation_visualize.py to generate heatmaps and curves")


if __name__ == "__main__":
    main()



