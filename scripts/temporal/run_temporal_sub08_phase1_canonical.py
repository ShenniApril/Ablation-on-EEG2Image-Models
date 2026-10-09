import argparse
import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

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

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import ttest_1samp

# 添加上级目录以导入其他模块
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 将当前工作目录加入，以便导入 module
CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.insert(0, CWD)

# 引入 ablation_temporal_stft 中的核心功能
import ablation_temporal_stft as phase1
from statistical_testing import _multipletests_fdr_bh

# 修正变量名
FS = phase1.FS
SELECTED_CHANNELS = phase1.SELECTED_CHANNELS
TIME_WINDOWS = phase1.TIME_WINDOWS
FREQ_BANDS = phase1.FREQ_BANDS
STFT_NPERSEG = phase1.STFT_NPERSEG
STFT_NOVERLAP = phase1.STFT_NOVERLAP
STFT_NFFT = phase1.STFT_NFFT
build_experiment_list = phase1.build_experiment_list
run_experiment = phase1.run_experiment

def build_dataset(subject_id: int, eeg_data_dir: Path, image_feature_dir: Path, aug_feature_dir: Path):
    from module.dataset import EEGPreImageDataset
    return EEGPreImageDataset(
        subject_ids=[subject_id],
        eeg_data_dir=str(eeg_data_dir),
        selected_channels=SELECTED_CHANNELS,
        time_window=[0, 250],
        image_feature_dir=str(image_feature_dir),
        text_feature_dir="",
        image_aug=True,
        aug_image_feature_dirs=[str(aug_feature_dir)],
        average=True,
        _random=False,
        eeg_transform=None,
        train=False,
        image_test_aug=True,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
    )

def load_models(checkpoint_path: Path, input_feature_dim: int, device: torch.device):
    from module.eeg_encoder.model import EEGProject
    from module.projector import ProjectorLinear
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "state_dict" in checkpoint and "model_state_dict" not in checkpoint:
        ckpt_data = checkpoint['state_dict']
    else:
        ckpt_data = checkpoint

    projector_weight = ckpt_data["eeg_projector_state_dict"]["linear.weight"]
    output_feature_dim, _ = projector_weight.shape

    model = EEGProject(
        feature_dim=input_feature_dim,
        eeg_sample_points=250,
        channels_num=len(SELECTED_CHANNELS),
    ).to(device)

    eeg_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)
    image_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)

    model.load_state_dict(ckpt_data["model_state_dict"])
    eeg_projector.load_state_dict(ckpt_data["eeg_projector_state_dict"])
    image_projector.load_state_dict(ckpt_data["img_projector_state_dict"])

    for net in (model, eeg_projector, image_projector):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return model, eeg_projector, image_projector

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 STFT Canonical Runner (5 Seeds)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["priority", "standard", "full"], default="full")
    
    # Paths assuming running from NeuroBridge folder
    default_data_dir = Path("data")
    parser.add_argument("--base-ckpt-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--eeg-data-dir", type=Path, default=REPO_ROOT / "data" / "things_eeg" / "preprocessed_eeg")
    parser.add_argument("--image-feature-dir", type=Path, default=REPO_ROOT / "data" / "things_eeg" / "image_feature" / "RN50")
    parser.add_argument("--aug-feature-dir", type=Path, default=REPO_ROOT / "data" / "things_eeg" / "image_feature" / "RN50" / "GaussianBlur-GaussianNoise-LowResolution-Mosaic")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "temporal_sub08_phase1_canonical")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Running Phase 1 Canonical STFT Ablation (5 seeds) | device={device} | mode={args.mode}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5 Seeds configuration
    seeds = [42, 3407, 1234, 2025, 9999]
    sub_id = 8
    sub_name = f"sub-{sub_id:02d}"
    
    # 生成实验列表
    # 我们这里需要全部的 time-frequency 组合，包括所有的 low_gamma / high_gamma
    # 如果 mode 是 full，build_experiment_list 会自动生成基于 PRIORITY_MATRIX 的列表。
    experiments = phase1.build_experiment_list(args.mode)
    
    all_seed_results = defaultdict(list)
    baseline_metrics_seeds = {}

    # Load Data (same for all seeds of sub-08)
    dataset = build_dataset(sub_id, args.eeg_data_dir, args.image_feature_dir, args.aug_feature_dir)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    original_eeg = batch[0].numpy().astype(np.float32)
    raw_img_feats = batch[1].to(device)
    correct_cols = torch.arange(len(raw_img_feats), dtype=torch.long)
    
    for seed in seeds:
        ckpt_path = args.base_ckpt_dir / f"intra-subjects-seed{seed}" / f"{sub_name}-seed{seed}" / "checkpoint_test_best.pth"
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint for seed {seed} not found at {ckpt_path}. Skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing Seed {seed}")
        print(f"{'='*60}")
        
        # Load Model
        model, eeg_projector, image_projector = load_models(ckpt_path, raw_img_feats.shape[-1], device)
        with torch.inference_mode():
            projected_images = image_projector(raw_img_feats).cpu()
            
        # Run experiments for this seed
        baseline_metrics = None
        for i, exp in enumerate(experiments):
            print(f"[{i+1:02d}/{len(experiments):02d}] {exp['name']} ...", end=" ", flush=True)
            result = run_experiment(
                exp, original_eeg, model, eeg_projector,
                projected_images, correct_cols, device, baseline_metrics
            )
            if exp["name"] == "baseline":
                baseline_metrics = result
                baseline_metrics_seeds[seed] = result
            
            all_seed_results[exp["name"]].append(result)
            
            drop_str = ""
            if result["top1_drop"] is not None:
                drop_str = f"  Δtop1={result['top1_drop']:+.4f}"
            print(f"Top-1={result['top1']:.4f}{drop_str}")

    print(f"\n{'='*60}")
    print("Averaging across 5 seeds and performing 1-sample t-test...")
    
    averaged_results = []
    pvals = []
    cond_names_for_fdr = []
    
    for exp in experiments:
        name = exp["name"]
        seed_res = all_seed_results[name]
        
        if not seed_res:
            continue
            
        avg_top1 = float(np.mean([r["top1"] for r in seed_res]))
        avg_top5 = float(np.mean([r["top5"] for r in seed_res]))
        
        if name == "baseline":
            averaged_results.append({
                "name": name,
                "time_window": exp.get("time_window"),
                "freq_band": exp.get("freq_band"),
                "priority": exp.get("priority", "control"),
                "top1": avg_top1,
                "top5": avg_top5,
                "top1_drop": None,
                "top5_drop": None
            })
            continue
            
        # Extract top1_drop across seeds
        drops = [r["top1_drop"] for r in seed_res if r["top1_drop"] is not None]
        avg_top1_drop = float(np.mean(drops)) if drops else None
        avg_top5_drop = float(np.mean([r["top5_drop"] for r in seed_res if r["top5_drop"] is not None])) if drops else None
        
        averaged_results.append({
            "name": name,
            "time_window": exp.get("time_window"),
            "freq_band": exp.get("freq_band"),
            "priority": exp.get("priority", "control"),
            "top1": avg_top1,
            "top5": avg_top5,
            "top1_drop": avg_top1_drop,
            "top5_drop": avg_top5_drop
        })
        
        # 1-sample t-test against 0 for top1_drop
        if len(drops) > 1:
            t_stat, p_val = ttest_1samp(drops, 0.0, alternative='greater')  # alternative='greater' if we expect drop > 0, or two-sided?
            # Use two-sided for general significant change
            t_stat, p_val = ttest_1samp(drops, 0.0)
            pvals.append(p_val)
            cond_names_for_fdr.append(name)
            
    # FDR Correction
    reject, pvals_corrected, _, _ = _multipletests_fdr_bh(pvals, alpha=0.05)
    
    # Save averaged CSV
    csv_path = args.output_dir / "stft_ablation_results.csv"
    fieldnames = ["name", "time_window", "freq_band", "priority", "top1", "top5", "top1_drop", "top5_drop"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(averaged_results)
        
    # Save FDR stats
    fdr_stats = []
    for name, p_raw, p_fdr, is_sig in zip(cond_names_for_fdr, pvals, pvals_corrected, reject):
        fdr_stats.append({
            "Condition": name,
            "p_value_raw": p_raw,
            "p_value_fdr": p_fdr,
            "Significant_FDR_0.05": is_sig
        })
        
    fdr_csv_path = args.output_dir / "mcnemar_fdr_results.csv"
    fdr_fieldnames = ["Condition", "p_value_raw", "p_value_fdr", "Significant_FDR_0.05"]
    with fdr_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fdr_fieldnames)
        writer.writeheader()
        writer.writerows(fdr_stats)

    print(f"\nSaved averaged results to {csv_path}")
    print(f"Saved FDR stats to {fdr_csv_path}")

if __name__ == "__main__":
    main()
