"""Group Phase 1 DC-preserving temporal-window masking evaluation.

Evaluates two model settings across subjects and training seeds. Per-run JSON
files make the evaluation resumable; summary CSVs use subjects as the
group-level statistical units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.signal import istft, stft
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
SETTINGS = ("intra-subjects", "inter-subjects")
SEEDS_DEFAULT = (1234, 2025, 42, 9999, 3407)
WINDOWS = {
    "T0": (0, 13),
    "T1": (13, 38),
    "T2": (38, 75),
    "T3": (75, 125),
    "T4": (125, 200),
}
WINDOW_KEYS = {
    "T0": "T0_0-52ms",
    "T1": "T1_52-152ms",
    "T2": "T2_152-300ms",
    "T3": "T3_300-500ms",
    "T4": "T4_500-800ms",
}
SELECTED_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
FS = 250.0
STFT_NPERSEG = 32
STFT_NOVERLAP = 16
STFT_NFFT = 64
ALGORITHM_VERSION = "dc-preserving-full-frequency-window-v1"
CONDITIONS = ("baseline", "T0", "T1", "T2", "T3", "T4")


def _add_import_paths(neurobridge_root: Path) -> None:
    paths = (neurobridge_root, SCRIPT_DIR)
    for path in paths:
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stft_mask_full_freq_window(eeg: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Mask non-DC components within one time window, preserving outside data."""
    eeg = np.asarray(eeg, dtype=np.float32)
    original_shape = eeg.shape
    start, end = window
    flat = eeg.reshape(-1, eeg.shape[-1])
    segment = flat[:, start:end]

    # Both T0 (13 samples) and T1 (25 samples) are shorter than nperseg=32.
    if segment.shape[-1] < STFT_NPERSEG:
        dc = segment.mean(axis=-1, keepdims=True)
        ablated = np.broadcast_to(dc, segment.shape).copy().astype(np.float32)
        result = flat.copy()
        result[:, start:end] = ablated
        return result.reshape(original_shape)

    _, _, spectrum = stft(
        segment,
        fs=FS,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        axis=-1,
    )
    masked = np.zeros_like(spectrum)
    masked[:, 0, :] = spectrum[:, 0, :]
    _, reconstructed = istft(
        masked,
        fs=FS,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        time_axis=-1,
        freq_axis=-2,
    )
    segment_length = end - start
    if reconstructed.shape[-1] >= segment_length:
        reconstructed = reconstructed[:, :segment_length]
    else:
        reconstructed = np.pad(
            reconstructed,
            ((0, 0), (0, segment_length - reconstructed.shape[-1])),
            mode="edge",
        )
    result = flat.copy()
    result[:, start:end] = reconstructed.astype(np.float32)
    return result.reshape(original_shape)


def _retrieval_metrics(
    eeg: np.ndarray,
    model: torch.nn.Module,
    eeg_projector: torch.nn.Module,
    image_projector: torch.nn.Module,
    image_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start:start + batch_size]).to(device)
            outputs.append(eeg_projector(model(batch)).cpu())
        eeg_features = torch.cat(outputs, dim=0)
        projected_images = image_projector(image_features.to(device)).cpu()

    eeg_features = torch.nn.functional.normalize(eeg_features, p=2, dim=1)
    projected_images = torch.nn.functional.normalize(projected_images, p=2, dim=1)
    ranking = (eeg_features @ projected_images.T).argsort(dim=1, descending=True)
    correct = torch.arange(len(projected_images), dtype=torch.long)
    ranks = (ranking == correct[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def _checkpoint_path(root: Path, setting: str, subject: int, seed: int) -> Path:
    base = root / f"{setting}-seed{seed}"
    subject_label = f"{subject:02d}"
    candidates = (
        base / f"sub-{subject_label}-seed{seed}" / "checkpoint_test_best.pth",
        base / f"inter-leave-sub-{subject_label}-seed{seed}" / "checkpoint_test_best.pth",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_models(
    setting: str, checkpoint_path: Path, feature_dim: int, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    from module.eeg_encoder.model import EEGProject, TSConv
    from module.projector import ProjectorLinear

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    projector_weight = checkpoint["eeg_projector_state_dict"]["linear.weight"]
    output_dim = projector_weight.shape[0]
    if setting == "intra-subjects":
        model = EEGProject(
            feature_dim=feature_dim, eeg_sample_points=250, channels_num=17
        ).to(device)
    else:
        model = TSConv(
            feature_dim=feature_dim, eeg_sample_points=250, channels_num=63
        ).to(device)
    eeg_projector = ProjectorLinear(feature_dim, output_dim).to(device)
    image_projector = ProjectorLinear(feature_dim, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])

    for module in (model, eeg_projector, image_projector):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, eeg_projector, image_projector


def _dataset_for_subject(
    setting: str, subject: int, data_root: Path
) -> tuple[np.ndarray, torch.Tensor]:
    from module.dataset import EEGPreImageDataset

    channels = SELECTED_CHANNELS if setting == "intra-subjects" else []
    image_dir = data_root / "image_feature" / "RN50"
    aug_dir = image_dir / "GaussianBlur-GaussianNoise-LowResolution-Mosaic"
    dataset = EEGPreImageDataset(
        subject_ids=[subject],
        eeg_data_dir=str(data_root / "preprocessed_eeg"),
        selected_channels=channels,
        time_window=[0, 250],
        image_feature_dir=str(image_dir),
        text_feature_dir="",
        image_aug=True,
        aug_image_feature_dirs=[str(aug_dir)],
        average=True,
        _random=False,
        eeg_transform=None,
        train=False,
        image_test_aug=True,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
    )
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    return batch[0].numpy().astype(np.float32), batch[1]


def _artifact_path(
    output_dir: Path, setting: str, subject: int, seed: int, condition: str
) -> Path:
    name = f"{setting}_sub{subject:02d}_seed{seed}_{condition}.json"
    return output_dir / "runs" / name


def _checkpoint_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "checkpoint_id": path.resolve().relative_to(root.resolve()).as_posix(),
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }


def _load_cached(
    path: Path, expected: dict[str, Any], resume: bool
) -> dict[str, Any] | None:
    if not resume or not path.is_file():
        return None
    payload = _read_json(path)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Cached result does not match current input ({key}): {path}"
            )
    if payload.get("status") == "complete":
        return payload
    return None


def _evaluate(
    setting: str,
    subject: int,
    seed: int,
    condition: str,
    eeg: np.ndarray,
    image_features: torch.Tensor,
    model: torch.nn.Module,
    eeg_projector: torch.nn.Module,
    image_projector: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if condition == "baseline":
        condition_eeg = eeg
    else:
        condition_eeg = stft_mask_full_freq_window(eeg, WINDOWS[condition])
    return _retrieval_metrics(
        condition_eeg,
        model,
        eeg_projector,
        image_projector,
        image_features,
        device,
        batch_size,
    )


def _bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        original_index = order[reverse_index]
        running = min(running, values[original_index] * count / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    output_dir: Path, checkpoint_root: Path | None = None
) -> dict[str, Any]:
    run_dir = output_dir / "runs"
    payloads = []
    if run_dir.exists():
        for path in sorted(run_dir.glob("*.json")):
            payload = _read_json(path)
            if payload.get("status") != "complete":
                continue
            changed = False
            if (
                "checkpoint_id" not in payload
                and payload.get("checkpoint_path")
                and checkpoint_root is not None
            ):
                payload["checkpoint_id"] = Path(
                    payload.pop("checkpoint_path")
                ).resolve().relative_to(checkpoint_root.resolve()).as_posix()
                changed = True
            actual_key = WINDOW_KEYS.get(
                payload["condition"], payload.get("time_window_key")
            )
            if payload.get("time_window_key") != actual_key:
                payload["time_window_key"] = actual_key
                changed = True
            if changed:
                _atomic_json(path, payload)
            payloads.append(payload)
    raw_rows: list[dict[str, Any]] = []
    for payload in payloads:
        raw_rows.append({
            "setting": payload["setting"],
            "subject": payload["subject"],
            "seed": payload["seed"],
            "condition": payload["condition"],
            "time_window_key": WINDOW_KEYS.get(
                payload["condition"], payload.get("time_window_key")
            ),
            "top1_baseline": payload["top1_baseline"],
            "top5_baseline": payload["top5_baseline"],
            "top1_ablated": payload["metrics"]["top1"],
            "top5_ablated": payload["metrics"]["top5"],
            "top1_drop": payload["top1_drop"],
            "top5_drop": payload["top5_drop"],
            "mean_rank": payload["metrics"]["mean_rank"],
            "median_rank": payload["metrics"]["median_rank"],
            "checkpoint_id": payload["checkpoint_id"],
        })
    raw_fields = list(raw_rows[0]) if raw_rows else [
        "setting", "subject", "seed", "condition", "time_window_key",
        "top1_baseline", "top5_baseline", "top1_ablated", "top5_ablated",
        "top1_drop", "top5_drop", "mean_rank", "median_rank", "checkpoint_id",
    ]
    _write_csv(output_dir / "per_seed_metrics.csv", raw_rows, raw_fields)

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["condition"] != "baseline":
            grouped.setdefault(
                (row["setting"], row["subject"], row["condition"]), []
            ).append(row)
    subject_rows = []
    for (setting, subject, condition), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        subject_rows.append({
            "setting": setting,
            "subject": subject,
            "condition": condition,
            "seed_count": len(rows),
            "mean_top1_baseline": float(np.mean([r["top1_baseline"] for r in rows])),
            "mean_top5_baseline": float(np.mean([r["top5_baseline"] for r in rows])),
            "mean_top1_ablated": float(np.mean([r["top1_ablated"] for r in rows])),
            "mean_top5_ablated": float(np.mean([r["top5_ablated"] for r in rows])),
            "mean_top1_drop": float(np.mean([r["top1_drop"] for r in rows])),
            "mean_top5_drop": float(np.mean([r["top5_drop"] for r in rows])),
        })
    subject_fields = list(subject_rows[0]) if subject_rows else [
        "setting", "subject", "condition", "seed_count",
        "mean_top1_baseline", "mean_top5_baseline", "mean_top1_ablated",
        "mean_top5_ablated", "mean_top1_drop", "mean_top5_drop",
    ]
    _write_csv(output_dir / "per_subject_condition.csv", subject_rows, subject_fields)

    summary_rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        p_values_by_metric: dict[str, list[float]] = {"top1": [], "top5": []}
        setting_rows = []
        for condition in WINDOWS:
            observations = [
                row for row in subject_rows
                if row["setting"] == setting
                and row["condition"] == condition
                and row["seed_count"] == len(SEEDS_DEFAULT)
            ]
            drops: dict[str, np.ndarray] = {
                metric: np.asarray(
                    [row[f"mean_{metric}_drop"] for row in observations],
                    dtype=np.float64,
                )
                for metric in ("top1", "top5")
            }
            row: dict[str, Any] = {
                "setting": setting,
                "condition": condition,
                "time_window_key": WINDOW_KEYS[condition],
                "subject_count": len(observations),
                "seed_count_per_subject": len(SEEDS_DEFAULT),
            }
            for metric in ("top1", "top5"):
                values = drops[metric]
                if len(values) >= 2:
                    mean = float(np.mean(values))
                    sd = float(np.std(values, ddof=1))
                    se = sd / math.sqrt(len(values))
                    critical = float(student_t.ppf(0.975, len(values) - 1))
                    ci_low, ci_high = mean - critical * se, mean + critical * se
                    p_value = (
                        1.0 if np.all(values == 0.0)
                        else float(ttest_1samp(values, 0.0).pvalue)
                    )
                elif len(values) == 1:
                    mean, sd, ci_low, ci_high, p_value = (
                        float(values[0]), float("nan"),
                        float("nan"), float("nan"), float("nan"),
                    )
                else:
                    mean, sd, ci_low, ci_high, p_value = (
                        float("nan"),) * 5
                row.update({
                    f"mean_{metric}_drop": mean,
                    f"sd_{metric}_drop": sd,
                    f"{metric}_ci95_low": ci_low,
                    f"{metric}_ci95_high": ci_high,
                    f"{metric}_p": p_value,
                })
                p_values_by_metric[metric].append(p_value)
            setting_rows.append(row)

        for metric in ("top1", "top5"):
            valid_indices = [
                index for index, value in enumerate(p_values_by_metric[metric])
                if np.isfinite(value)
            ]
            adjusted = [float("nan")] * len(WINDOWS)
            if valid_indices:
                values = [p_values_by_metric[metric][index] for index in valid_indices]
                q_values = _bh_adjust(values)
                for index, q_value in zip(valid_indices, q_values):
                    adjusted[index] = q_value
            for row, q_value in zip(setting_rows, adjusted):
                row[f"{metric}_q_bh_5windows"] = q_value
        summary_rows.extend(setting_rows)

    summary_fields = list(summary_rows[0]) if summary_rows else [
        "setting", "condition", "time_window_key", "subject_count",
        "seed_count_per_subject", "mean_top1_drop", "sd_top1_drop",
        "top1_ci95_low", "top1_ci95_high", "top1_p", "top1_q_bh_5windows",
        "mean_top5_drop", "sd_top5_drop", "top5_ci95_low", "top5_ci95_high",
        "top5_p", "top5_q_bh_5windows",
    ]
    _write_csv(output_dir / "group_summary.csv", summary_rows, summary_fields)

    expected_runs = len(SETTINGS) * 10 * len(SEEDS_DEFAULT) * len(CONDITIONS)
    counts: dict[str, int] = {}
    for setting in SETTINGS:
        counts[setting] = sum(row["setting"] == setting for row in raw_rows)
    metadata = {
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete" if len(raw_rows) == expected_runs else "incomplete",
        "expected_evaluations": expected_runs,
        "completed_evaluations": len(raw_rows),
        "completed_by_setting": counts,
        "accuracy_unit": "fraction; drops are baseline minus ablated accuracy",
        "group_unit": "subject; five seed-level paired drops averaged within subject",
        "inference": "two-sided one-sample t-test of subject-level paired drops against zero",
        "fdr": "Benjamini-Hochberg separately for Top-1 and Top-5 across five windows within each setting",
        "confidence_interval": "two-sided 95% Student-t interval across subjects",
        "sample_rate_hz": FS,
        "settings": list(SETTINGS),
        "subjects": list(range(1, 11)),
        "training_seeds": list(SEEDS_DEFAULT),
        "window_slices": {key: list(value) for key, value in WINDOWS.items()},
        "window_labels": WINDOW_KEYS,
        "evaluation": {
            "eeg_sample_range": [0, 250],
            "repetitions_averaged": True,
            "random_order": False,
            "image_test_augmentation": True,
            "eeg_test_augmentation": False,
            "intra_channels": SELECTED_CHANNELS,
            "inter_channels": "all 63 channels",
            "top1_top5": "paired retrieval against the same subject's image features",
            "drop": "same-checkpoint baseline accuracy minus masked accuracy",
            "checkpoint": "checkpoint_test_best.pth",
            "inference_batch_size": 200,
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        },
        "stft": {
            "nperseg": STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "nfft": STFT_NFFT,
            "short_window_fallback": "T0 and T1 are shorter than 32 samples; replace segment by its mean",
            "long_window_masking": "zero non-DC STFT coefficients and preserve the DC bin",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    _atomic_json(output_dir / "manifest.json", metadata)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint_root = WORKSPACE_ROOT / "results"
    default_output_dir = (
        SCRIPT_DIR / "results" / "phase1_window_group"
        if SCRIPT_DIR.name == "temporal_ablation"
        else WORKSPACE_ROOT / "results" / "temporal" / "phase1_window_group"
    )
    parser.add_argument(
        "--neurobridge-root", type=Path,
        default=WORKSPACE_ROOT,
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=WORKSPACE_ROOT / "data" / "things_eeg",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=default_checkpoint_root
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=default_output_dir,
    )
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS_DEFAULT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summary CSVs and manifest from completed per-condition JSONs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if len(set(args.subjects)) != len(args.subjects) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Subjects and seeds must not contain duplicates")
    _add_import_paths(args.neurobridge_root)
    if args.summarize_only:
        metadata = _summarize(args.output_dir, args.checkpoint_root)
        print(
            f"{metadata['status']}: "
            f"{metadata['completed_evaluations']}/"
            f"{metadata['expected_evaluations']} evaluation rows"
        )
        return 0 if metadata["status"] == "complete" else 2

    checkpoints = {
        (setting, subject, seed): _checkpoint_path(
            args.checkpoint_root, setting, subject, seed
        )
        for setting in args.settings
        for subject in args.subjects
        for seed in args.seeds
    }
    missing = [key for key, path in checkpoints.items() if not path.is_file()]
    print(
        f"Checkpoint inventory: {len(checkpoints) - len(missing)}/"
        f"{len(checkpoints)} present"
    )
    if missing:
        for setting, subject, seed in missing:
            print(f"MISSING {setting} sub-{subject:02d} seed={seed}")
        if not args.dry_run:
            raise FileNotFoundError(
                "Checkpoint inventory is incomplete; no inference was started."
            )
        return 2
    if args.dry_run:
        print("Dry run passed; no model inference was performed.")
        return 0

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for setting in args.settings:
        for subject in args.subjects:
            print(f"\n[{setting}] subject {subject:02d}: loading evaluation data")
            eeg, image_features = _dataset_for_subject(
                setting, subject, args.data_root
            )
            expected_channels = 17 if setting == "intra-subjects" else 63
            if eeg.ndim != 3 or eeg.shape[1:] != (expected_channels, 250):
                raise ValueError(
                    f"Unexpected EEG shape for {setting}, sub-{subject:02d}: "
                    f"{eeg.shape}; expected (N,{expected_channels},250)"
                )
            if len(eeg) != len(image_features):
                raise ValueError("EEG and image-feature sample counts differ")

            for seed in args.seeds:
                checkpoint = checkpoints[(setting, subject, seed)]
                fingerprint = _checkpoint_fingerprint(
                    checkpoint, args.checkpoint_root
                )
                common = {
                    "schema_version": 1,
                    "algorithm_version": ALGORITHM_VERSION,
                    "setting": setting,
                    "subject": subject,
                    "seed": seed,
                    **fingerprint,
                }
                baseline_path = _artifact_path(
                    args.output_dir, setting, subject, seed, "baseline"
                )
                baseline_payload = _load_cached(
                    baseline_path, common, not args.no_resume
                )
                needed = []
                for condition in CONDITIONS:
                    path = _artifact_path(
                        args.output_dir, setting, subject, seed, condition
                    )
                    cached = _load_cached(path, common, not args.no_resume)
                    if condition == "baseline":
                        baseline_payload = cached or baseline_payload
                    if cached is None:
                        needed.append((condition, path))
                if not needed:
                    print(f"  seed {seed}: cached (all six conditions)")
                    continue

                model, eeg_projector, image_projector = _load_models(
                    setting,
                    checkpoint,
                    int(image_features.shape[-1]),
                    device,
                )
                baseline_metrics = (
                    baseline_payload["metrics"] if baseline_payload else None
                )
                for condition, artifact in needed:
                    if condition == "baseline" and baseline_metrics is not None:
                        continue
                    print(f"  seed {seed}, {condition}: evaluating", flush=True)
                    metrics = _evaluate(
                        setting,
                        subject,
                        seed,
                        condition,
                        eeg,
                        image_features,
                        model,
                        eeg_projector,
                        image_projector,
                        device,
                        args.batch_size,
                    )
                    if condition == "baseline":
                        baseline_metrics = metrics
                    if baseline_metrics is None:
                        raise RuntimeError("Baseline must be evaluated before ablations")
                    payload = {
                        **common,
                        "condition": condition,
                        "time_window_key": WINDOW_KEYS.get(condition),
                        "window_samples": (
                            list(WINDOWS[condition]) if condition in WINDOWS else None
                        ),
                        "status": "complete",
                        "metrics": metrics,
                        "top1_baseline": baseline_metrics["top1"],
                        "top5_baseline": baseline_metrics["top5"],
                        "top1_drop": (
                            0.0 if condition == "baseline"
                            else baseline_metrics["top1"] - metrics["top1"]
                        ),
                        "top5_drop": (
                            0.0 if condition == "baseline"
                            else baseline_metrics["top5"] - metrics["top5"]
                        ),
                    }
                    _atomic_json(artifact, payload)
                    print(
                        f"    Top-1={metrics['top1']:.4f} "
                        f"Top-5={metrics['top5']:.4f} "
                        f"drop={payload['top1_drop']:+.4f}",
                        flush=True,
                    )
                del model, eeg_projector, image_projector
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
            del eeg, image_features

    metadata = _summarize(args.output_dir, args.checkpoint_root)
    print(
        f"\n{metadata['status']}: "
        f"{metadata['completed_evaluations']}/{metadata['expected_evaluations']} "
        "evaluation rows"
    )
    print(f"Results: {args.output_dir}")
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
