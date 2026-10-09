import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import json

# Add the repository root to sys.path so module.* and local scripts resolve after cloning
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.model import EEGProject, TSConv
from module.projector import ProjectorLinear

from ablation_temporal_stft_v2b import (
    encode_eeg, retrieval_metrics, 
    build_experiment_list, run_experiment,
    FS, DEFAULT_DATA_DIR, TIME_WINDOWS, FREQ_BANDS,
    STFT_NPERSEG, STFT_NOVERLAP, STFT_NFFT, SELECTED_CHANNELS
)

def load_models_for_setting(setting, checkpoint_path, input_feature_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    projector_weight = checkpoint["eeg_projector_state_dict"]["linear.weight"]
    output_feature_dim, _ = projector_weight.shape

    if setting == 'intra-subjects':
        model = EEGProject(
            feature_dim=input_feature_dim,
            eeg_sample_points=250,
            channels_num=17,
        ).to(device)
    else:
        model = TSConv(
            feature_dim=input_feature_dim,
            eeg_sample_points=250,
            channels_num=63,
        ).to(device)

    eeg_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)
    image_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    
    model.eval()
    eeg_projector.eval()
    image_projector.eval()
    
    return model, eeg_projector, image_projector

def build_dataset_for_setting(setting, args):
    channels = SELECTED_CHANNELS if setting == 'intra-subjects' else []
    return EEGPreImageDataset(
        subject_ids=[args.subject],
        eeg_data_dir=str(args.eeg_data_dir),
        selected_channels=channels,
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
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generalization Evaluation: STFT Time-Frequency Masking Ablation across seeds"
    )
    parser.add_argument("--subject",    type=int,  default=8)
    parser.add_argument("--device",     type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int,  default=200)
    parser.add_argument(
        "--mode",
        choices=["priority", "standard", "full"],
        default="priority",
        help="priority: 只跑优先级≥4；standard: 优先级≥3；full: 全部组合"
    )
    REAL_DATA_DIR = REPO_ROOT / "data"
    parser.add_argument(
        "--eeg-data-dir",
        type=Path,
        default=REAL_DATA_DIR / "things_eeg" / "preprocessed_eeg",
    )
    parser.add_argument(
        "--image-feature-dir",
        type=Path,
        default=REAL_DATA_DIR / "things_eeg" / "image_feature" / "RN50",
    )
    parser.add_argument(
        "--aug-feature-dir",
        type=Path,
        default=REAL_DATA_DIR / "things_eeg" / "image_feature" / "RN50" / "GaussianBlur-GaussianNoise-LowResolution-Mosaic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "generalization_ablation",
    )
    parser.add_argument(
        "--base-results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root directory holding the per-seed checkpoint subdirectories",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"[Generalization] STFT Temporal Ablation | subject={args.subject} | device={device} | mode={args.mode}")

    experiments = build_experiment_list(args.mode)
    
    # 泛化评估所需的设定与种子列表
    settings = ['intra-subjects', 'inter-subjects']
    seeds = [1234, 2025, 42, 9999, 3407]
    base_results_dir = args.base_results_dir
    
    # 初始化累加结构
    accumulated_results = {setting: {exp['name']: [] for exp in experiments} for setting in settings}
    valid_runs = {setting: 0 for setting in settings}
    
    # 遍历每个设置及种子组合
    for setting in settings:
        print(f"\nLoading test EEG data for {setting}...")
        dataset = build_dataset_for_setting(setting, args)
        loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        batch   = next(iter(loader))

        original_eeg    = batch[0].numpy().astype(np.float32)
        raw_img_feats   = batch[1].to(device)
        correct_cols    = torch.arange(len(raw_img_feats), dtype=torch.long)

        for seed in seeds:
            # 动态构建检查点路径
            ckpt_path = base_results_dir / f"{setting}-seed{seed}" / f"sub-{args.subject:02d}-seed{seed}" / "checkpoint_test_best.pth"
            
            # 兼容跨被试实验目录名为 inter-leave-sub-xx 的情况
            if not ckpt_path.exists():
                if setting == 'inter-subjects':
                    ckpt_path_alt = base_results_dir / f"{setting}-seed{seed}" / f"inter-leave-sub-{args.subject:02d}-seed{seed}" / "checkpoint_test_best.pth"
                    if ckpt_path_alt.exists():
                        ckpt_path = ckpt_path_alt

            if not ckpt_path.exists():
                print(f"[Warning] Checkpoint not found: {ckpt_path}")
                continue
                
            print(f"\n>>> Running for {setting} - Seed {seed} <<<")
            print(f"Loading checkpoint: {ckpt_path}")
            
            try:
                model, eeg_projector, image_projector = load_models_for_setting(setting, ckpt_path, raw_img_feats.shape[-1], device)
            except Exception as e:
                print(f"[Error] Failed to load checkpoint: {e}")
                continue
                
            with torch.inference_mode():
                projected_images = image_projector(raw_img_feats).cpu()
                
            baseline_metrics = None
            
            # 对单个种子执行全部条件推理
            for i, exp in enumerate(experiments):
                print(f"  [{i+1:02d}/{len(experiments):02d}] {exp['name']} ...", end=" ", flush=True)
                result = run_experiment(
                    exp, original_eeg, model, eeg_projector,
                    projected_images, correct_cols, device, baseline_metrics
                )
                if exp["name"] == "baseline":
                    baseline_metrics = result
                
                accumulated_results[setting][exp['name']].append(result)
                print(f"Top-1={result['top1']:.4f}")
            
            valid_runs[setting] += 1

    for setting in settings:
        if valid_runs[setting] == 0:
            print(f"\n[Warning] No valid checkpoints found for {setting}.")
            continue

        # 平均结果计算
        print(f"\n=== Averaging metrics across {valid_runs[setting]} runs for {setting} ===")
        final_results = []
        
        for exp in experiments:
            name = exp['name']
            runs = accumulated_results[setting][name]
            if not runs:
                continue
            
            avg_top1 = np.mean([r['top1'] for r in runs])
            avg_top5 = np.mean([r['top5'] for r in runs])
            avg_mean_rank = np.mean([r['mean_rank'] for r in runs])
            avg_median_rank = np.mean([r['median_rank'] for r in runs])
            
            # 相对 Baseline 计算 Accuracy Drop
            if name == "baseline":
                avg_top1_drop = None
                avg_top5_drop = None
            else:
                baseline_runs = accumulated_results[setting]["baseline"]
                drops1 = [br['top1'] - r['top1'] for br, r in zip(baseline_runs, runs)]
                drops5 = [br['top5'] - r['top5'] for br, r in zip(baseline_runs, runs)]
                avg_top1_drop = np.mean(drops1)
                avg_top5_drop = np.mean(drops5)
                
            final_results.append({
                "name": name,
                "time_window": exp.get("time_window"),
                "freq_band": exp.get("freq_band"),
                "priority": exp.get("priority", "control"),
                "top1": round(avg_top1, 4),
                "top5": round(avg_top5, 4),
                "mean_rank": round(avg_mean_rank, 4),
                "median_rank": round(avg_median_rank, 4),
                "top1_drop": round(avg_top1_drop, 4) if avg_top1_drop is not None else None,
                "top5_drop": round(avg_top5_drop, 4) if avg_top5_drop is not None else None,
            })
            
        # 保存 CSV
        csv_path = args.output_dir / f"generalization_stft_ablation_{setting}_sub{args.subject:02d}.csv"
        fieldnames = ["name", "time_window", "freq_band", "priority",
                      "top1", "top5", "mean_rank", "median_rank", "top1_drop", "top5_drop"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_results)
        print(f"\nSaved Averaged CSV -> {csv_path}")

if __name__ == "__main__":
    main()
