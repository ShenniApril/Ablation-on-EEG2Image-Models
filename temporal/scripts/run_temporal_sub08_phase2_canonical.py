from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


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
from torch.utils.data import DataLoader

import ablation_temporal_amplitude as phase2

try:
    import statistical_testing
except ModuleNotFoundError:
    statistical_testing = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "temporal_sub08_phase2_canonical"

CANONICAL_SUBJECT = 8
CANONICAL_SEEDS = [42, 3407, 9999, 2025, 1234]
CANONICAL_SETTINGS = ["intra-subjects-averaged", "inter-subjects-averaged"]
PAPER_FIGURE_SETTING = "intra-subjects-averaged"

CSV_FIELDS = [
    "setting",
    "subject",
    "seed",
    "checkpoint_path",
    "condition",
    "perturbation",
    "param_name",
    "param_value",
    "top1",
    "top5",
    "mean_rank",
    "median_rank",
    "top1_drop",
    "top5_drop",
]

NUMERIC_RESULT_FIELDS = ["top1", "top5", "mean_rank", "median_rank", "top1_drop", "top5_drop"]
BASELINE_NUMERIC_FIELDS = ["top1", "top5", "mean_rank", "median_rank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical temporal_sub08 Phase 2 pipeline using independent seed checkpoints."
    )
    parser.add_argument("--subject", type=int, default=CANONICAL_SUBJECT)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=CANONICAL_SETTINGS,
        help="Canonical settings to process, e.g. intra-subjects-averaged inter-subjects-averaged.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=phase2.DEFAULT_TOP_N)
    parser.add_argument("--phase1-csv", type=Path, default=phase2.DEFAULT_PHASE1_CSV)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--eeg-data-dir", type=Path, default=phase2.DEFAULT_DATA / "things_eeg" / "preprocessed_eeg")
    parser.add_argument("--image-feature-dir", type=Path, default=phase2.DEFAULT_DATA / "things_eeg" / "image_feature" / "RN50")
    parser.add_argument(
        "--aug-feature-dir",
        type=Path,
        default=phase2.DEFAULT_DATA / "things_eeg" / "image_feature" / "RN50" / "GaussianBlur-GaussianNoise-LowResolution-Mosaic",
    )
    return parser.parse_args()


def build_phase2_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return build_phase2_args_for_channels(cli_args, phase2.SELECTED_CHANNELS)


def build_phase2_args_for_channels(cli_args: argparse.Namespace, selected_channels: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        subject=cli_args.subject,
        eeg_data_dir=cli_args.eeg_data_dir,
        image_feature_dir=cli_args.image_feature_dir,
        aug_feature_dir=cli_args.aug_feature_dir,
        selected_channels=selected_channels,
    )


def infer_selected_channels_for_setting(cli_args: argparse.Namespace, setting: str) -> list[str]:
    checkpoint = resolve_seed_checkpoint(cli_args.results_root, setting, cli_args.subject, cli_args.seeds[0])
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    model_state_dict = checkpoint_data["model_state_dict"]
    if "model.0.weight" in model_state_dict:
        channel_count = int(model_state_dict["model.0.weight"].shape[1] // 250)
    elif "tsconv.4.weight" in model_state_dict:
        channel_count = int(model_state_dict["tsconv.4.weight"].shape[2])
    else:
        raise KeyError(f"Unable to infer channel count from checkpoint: {checkpoint}")

    if channel_count == len(phase2.SELECTED_CHANNELS):
        return list(phase2.SELECTED_CHANNELS)
    if channel_count == 63:
        return []
    raise ValueError(f"Unsupported channel count inferred from {checkpoint}: {channel_count}")


def load_dataset_bundle(cli_args: argparse.Namespace, device: torch.device, selected_channels: list[str]) -> dict[str, Any]:
    phase2_args = build_phase2_args_for_channels(cli_args, selected_channels)
    dataset = phase2.build_dataset(phase2_args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    raw_eeg = batch[0].numpy().astype(np.float32)
    raw_img_feats = batch[1].to(device)
    object_indices = batch[4].numpy()
    image_indices = batch[5].numpy()
    gallery_keys = list(zip(object_indices.tolist(), image_indices.tolist()))
    key_to_col = {key: idx for idx, key in enumerate(gallery_keys)}
    correct_cols = torch.tensor([key_to_col[key] for key in gallery_keys], dtype=torch.long)
    return {
        "raw_eeg": raw_eeg,
        "raw_img_feats": raw_img_feats,
        "correct_cols": correct_cols,
    }


def resolve_seed_checkpoint(results_root: Path, averaged_setting: str, subject: int, seed: int) -> Path:
    seed_setting = averaged_setting.replace("-averaged", "")
    checkpoint = results_root / f"{seed_setting}-seed{seed}" / f"sub-{subject:02d}-seed{seed}" / "checkpoint_test_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing seed checkpoint: {checkpoint}")
    return checkpoint


def resolve_averaged_checkpoint_reference(results_root: Path, averaged_setting: str, subject: int) -> dict[str, Any]:
    checkpoint = results_root / averaged_setting / f"sub-{subject:02d}" / "checkpoint_test_best.pth"
    reference: dict[str, Any] = {
        "path": str(checkpoint),
        "exists": checkpoint.exists(),
        "role": "reference_only",
        "usage": "provenance_or_cross_check_only",
    }
    if checkpoint.exists():
        stat = checkpoint.stat()
        reference["size_bytes"] = stat.st_size
        reference["last_modified"] = stat.st_mtime
    return reference


def build_seed_provenance(results_root: Path, averaged_setting: str, subject: int, seed: int, conditions: list[str]) -> dict[str, Any]:
    return {
        "canonical_subject": subject,
        "canonical_seed": seed,
        "canonical_seed_set": CANONICAL_SEEDS,
        "setting": averaged_setting,
        "result_layer_average_target": str(results_root / averaged_setting / f"sub-{subject:02d}" / "checkpoint_test_best.pth"),
        "seed_checkpoint_path": str(resolve_seed_checkpoint(results_root, averaged_setting, subject, seed)),
        "averaged_checkpoint_reference": resolve_averaged_checkpoint_reference(results_root, averaged_setting, subject),
        "conditions": conditions,
        "methodology": "Run Phase 2 on each seed checkpoint independently, then average result tables across seeds.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_seed_phase2(
    cli_args: argparse.Namespace,
    device: torch.device,
    dataset_bundle: dict[str, Any],
    setting: str,
    seed: int,
    conditions: list[str],
) -> dict[str, Any]:
    checkpoint = resolve_seed_checkpoint(cli_args.results_root, setting, cli_args.subject, seed)
    provenance = build_seed_provenance(cli_args.results_root, setting, cli_args.subject, seed, conditions)

    model, eeg_proj, img_proj = phase2.load_models(checkpoint, dataset_bundle["raw_img_feats"].shape[-1], device)
    with torch.inference_mode():
        projected_images = img_proj(dataset_bundle["raw_img_feats"]).cpu()

    baseline_feats = phase2.encode_eeg(dataset_bundle["raw_eeg"], model, eeg_proj, device, batch_size=cli_args.batch_size)
    baseline_metrics = phase2.retrieval_metrics(baseline_feats, projected_images, dataset_bundle["correct_cols"])

    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=seed)

    for condition in conditions:
        time_window_key, freq_band_key = phase2.parse_condition(condition)
        time_window = phase2.TIME_WINDOWS[time_window_key]
        freq_band = phase2.FREQ_BANDS[freq_band_key]

        for alpha in phase2.ALPHA_LEVELS:
            ablated = phase2.amplitude_scaling(dataset_bundle["raw_eeg"], time_window, freq_band, alpha)
            metrics = phase2.retrieval_metrics(
                phase2.encode_eeg(ablated, model, eeg_proj, device, batch_size=cli_args.batch_size),
                projected_images,
                dataset_bundle["correct_cols"],
            )
            results.append(
                {
                    "setting": setting,
                    "subject": cli_args.subject,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "condition": condition,
                    "perturbation": "scaling",
                    "param_name": "alpha",
                    "param_value": alpha,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "mean_rank": metrics["mean_rank"],
                    "median_rank": metrics["median_rank"],
                    "top1_drop": round(baseline_metrics["top1"] - metrics["top1"], 4),
                    "top5_drop": round(baseline_metrics["top5"] - metrics["top5"], 4),
                    "raw_preds": metrics["raw_preds"],
                }
            )

        for rand_ratio in phase2.PHASE_RAND_LEVELS:
            ablated = phase2.phase_randomization(dataset_bundle["raw_eeg"], time_window, freq_band, rand_ratio, rng)
            metrics = phase2.retrieval_metrics(
                phase2.encode_eeg(ablated, model, eeg_proj, device, batch_size=cli_args.batch_size),
                projected_images,
                dataset_bundle["correct_cols"],
            )
            results.append(
                {
                    "setting": setting,
                    "subject": cli_args.subject,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "condition": condition,
                    "perturbation": "phase_rand",
                    "param_name": "rand_ratio",
                    "param_value": rand_ratio,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "mean_rank": metrics["mean_rank"],
                    "median_rank": metrics["median_rank"],
                    "top1_drop": round(baseline_metrics["top1"] - metrics["top1"], 4),
                    "top5_drop": round(baseline_metrics["top5"] - metrics["top5"], 4),
                    "raw_preds": metrics["raw_preds"],
                }
            )

        for snr_db in phase2.SNR_LEVELS_DB:
            ablated = phase2.gaussian_noise_injection(dataset_bundle["raw_eeg"], time_window, freq_band, snr_db, rng)
            metrics = phase2.retrieval_metrics(
                phase2.encode_eeg(ablated, model, eeg_proj, device, batch_size=cli_args.batch_size),
                projected_images,
                dataset_bundle["correct_cols"],
            )
            results.append(
                {
                    "setting": setting,
                    "subject": cli_args.subject,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "condition": condition,
                    "perturbation": "gaussian_noise",
                    "param_name": "snr_db",
                    "param_value": snr_db,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "mean_rank": metrics["mean_rank"],
                    "median_rank": metrics["median_rank"],
                    "top1_drop": round(baseline_metrics["top1"] - metrics["top1"], 4),
                    "top5_drop": round(baseline_metrics["top5"] - metrics["top5"], 4),
                    "raw_preds": metrics["raw_preds"],
                }
            )

    return {
        "seed": seed,
        "checkpoint_path": str(checkpoint),
        "provenance": provenance,
        "baseline": baseline_metrics,
        "results": results,
    }


def save_seed_outputs(setting_dir: Path, seed_payload: dict[str, Any]) -> dict[str, Path]:
    seed_dir = setting_dir / "seeds" / f"seed-{seed_payload['seed']}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = seed_dir / "phase2_seed_results.csv"
    write_csv(csv_path, seed_payload["results"], CSV_FIELDS)

    json_path = seed_dir / "phase2_seed_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "temporal_sub08 Phase 2 canonical seed-level run",
                "subject": seed_payload["provenance"]["canonical_subject"],
                "setting": seed_payload["provenance"]["setting"],
                "seed": seed_payload["seed"],
                "provenance": seed_payload["provenance"],
                "baseline": seed_payload["baseline"],
                "results": seed_payload["results"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    provenance_path = seed_dir / "seed_provenance.json"
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(seed_payload["provenance"], handle, ensure_ascii=False, indent=2)

    return {"csv": csv_path, "json": json_path, "provenance": provenance_path}


def aggregate_seed_payloads(setting: str, subject: int, seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        "seed_count": len(seed_payloads),
        "raw_preds": [],
    }
    for field in BASELINE_NUMERIC_FIELDS:
        baseline[field] = float(np.mean([payload["baseline"][field] for payload in seed_payloads]))
    for payload in seed_payloads:
        baseline["raw_preds"].extend(bool(x) for x in payload["baseline"]["raw_preds"])

    grouped_rows: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for payload in seed_payloads:
        for row in payload["results"]:
            key = (
                row["condition"],
                row["perturbation"],
                row["param_name"],
                float(row["param_value"]),
            )
            grouped_rows[key].append(row)

    averaged_rows: list[dict[str, Any]] = []
    for key, rows in grouped_rows.items():
        exemplar = rows[0]
        averaged_row: dict[str, Any] = {
            "setting": setting,
            "subject": subject,
            "seed": "avg",
            "checkpoint_path": "",
            "condition": exemplar["condition"],
            "perturbation": exemplar["perturbation"],
            "param_name": exemplar["param_name"],
            "param_value": float(exemplar["param_value"]),
            "seed_count": len(rows),
            "source_seeds": [row["seed"] for row in rows],
            "raw_preds": [],
        }
        for field in NUMERIC_RESULT_FIELDS:
            averaged_row[field] = round(float(np.mean([float(row[field]) for row in rows])), 6)
        for row in rows:
            averaged_row["raw_preds"].extend(bool(x) for x in row["raw_preds"])
        averaged_rows.append(averaged_row)

    averaged_rows.sort(key=lambda row: (row["condition"], row["perturbation"], float(row["param_value"])))
    return {"baseline": baseline, "results": averaged_rows}


def save_averaged_outputs(
    cli_args: argparse.Namespace,
    setting_dir: Path,
    setting: str,
    conditions: list[str],
    seed_payloads: list[dict[str, Any]],
) -> dict[str, Path]:
    averaged = aggregate_seed_payloads(setting, cli_args.subject, seed_payloads)

    csv_path = setting_dir / "phase2_canonical_averaged.csv"
    averaged_csv_rows = []
    for row in averaged["results"]:
        averaged_csv_rows.append(
            {
                "setting": row["setting"],
                "subject": row["subject"],
                "seed": row["seed"],
                "checkpoint_path": row["checkpoint_path"],
                "condition": row["condition"],
                "perturbation": row["perturbation"],
                "param_name": row["param_name"],
                "param_value": row["param_value"],
                "top1": row["top1"],
                "top5": row["top5"],
                "mean_rank": row["mean_rank"],
                "median_rank": row["median_rank"],
                "top1_drop": row["top1_drop"],
                "top5_drop": row["top5_drop"],
            }
        )
    write_csv(csv_path, averaged_csv_rows, CSV_FIELDS)

    provenance = {
        "canonical_subject": cli_args.subject,
        "canonical_seeds": cli_args.seeds,
        "setting": setting,
        "conditions": conditions,
        "averaged_checkpoint_reference": resolve_averaged_checkpoint_reference(cli_args.results_root, setting, cli_args.subject),
        "methodology": "Independent seed checkpoints were evaluated first; averaged CSV/JSON were computed by arithmetic mean across seed result tables.",
        "seed_runs": [
            {
                "seed": payload["seed"],
                "checkpoint_path": payload["checkpoint_path"],
                "seed_provenance_path": str(setting_dir / "seeds" / f"seed-{payload['seed']}" / "seed_provenance.json"),
                "seed_csv_path": str(setting_dir / "seeds" / f"seed-{payload['seed']}" / "phase2_seed_results.csv"),
                "seed_json_path": str(setting_dir / "seeds" / f"seed-{payload['seed']}" / "phase2_seed_results.json"),
            }
            for payload in seed_payloads
        ],
    }

    json_path = setting_dir / "phase2_canonical_averaged.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "temporal_sub08 Phase 2 canonical averaged pipeline",
                "subject": cli_args.subject,
                "setting": setting,
                "conditions": conditions,
                "perturbation_types": ["scaling", "phase", "noise"],
                "params": {
                    "alpha_levels": phase2.ALPHA_LEVELS,
                    "phase_rand_levels": phase2.PHASE_RAND_LEVELS,
                    "snr_levels_db": phase2.SNR_LEVELS_DB,
                },
                "provenance": provenance,
                "aggregation": {
                    "numeric_fields": "arithmetic_mean_across_seeds",
                    "raw_preds": "concatenated_across_seeds_for_provenance_and_statistical_testing",
                },
                "baseline": averaged["baseline"],
                "results": averaged["results"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    provenance_path = setting_dir / "seed_provenance.json"
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)

    stats_path = setting_dir / "mcnemar_fdr_results.csv"
    if statistical_testing is not None:
        statistical_testing.calculate_mcnemar_and_fdr(None, json_path, None, stats_path)
    else:
        stats_path = None

    output_paths = {"csv": csv_path, "json": json_path, "provenance": provenance_path}
    if stats_path is not None:
        output_paths["stats"] = stats_path
    return output_paths


def run_setting(
    cli_args: argparse.Namespace,
    device: torch.device,
    setting: str,
    conditions: list[str],
) -> dict[str, Any]:
    setting_dir = cli_args.output_root / setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    selected_channels = infer_selected_channels_for_setting(cli_args, setting)
    dataset_bundle = load_dataset_bundle(cli_args, device, selected_channels)

    seed_payloads: list[dict[str, Any]] = []
    seed_output_paths: dict[int, dict[str, str]] = {}
    for seed in cli_args.seeds:
        print(f"  [Seed {seed}] running {setting}")
        seed_payload = run_seed_phase2(cli_args, device, dataset_bundle, setting, seed, conditions)
        seed_paths = save_seed_outputs(setting_dir, seed_payload)
        seed_output_paths[seed] = {name: str(path) for name, path in seed_paths.items()}
        seed_payloads.append(seed_payload)

    averaged_paths = save_averaged_outputs(cli_args, setting_dir, setting, conditions, seed_payloads)
    return {
        "setting_dir": setting_dir,
        "seed_outputs": seed_output_paths,
        "averaged_outputs": {name: str(path) for name, path in averaged_paths.items()},
    }


def write_manifest(cli_args: argparse.Namespace, conditions: list[str], outputs_by_setting: dict[str, dict[str, Any]]) -> Path:
    serialized_outputs: dict[str, dict[str, Any]] = {}
    for setting, payload in outputs_by_setting.items():
        serialized_outputs[setting] = {
            "setting_dir": str(payload["setting_dir"]),
            "seed_outputs": payload["seed_outputs"],
            "averaged_outputs": payload["averaged_outputs"],
        }
    manifest = {
        "subject": cli_args.subject,
        "canonical_seeds": cli_args.seeds,
        "conditions": conditions,
        "paper_figure_setting": PAPER_FIGURE_SETTING,
        "methodology": "5 independent seed checkpoints -> seed-level Phase 2 outputs -> result-layer average -> canonical averaged CSV/JSON",
        "averaged_checkpoint_policy": "Averaged checkpoints are reference-only and are not used as the primary Phase 2 result source.",
        "outputs": serialized_outputs,
    }
    manifest_path = cli_args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    conditions = phase2.load_top_conditions_from_phase1(args.phase1_csv, args.top_n)

    outputs_by_setting: dict[str, dict[str, Any]] = {}
    for setting in args.settings:
        print(f"[Canonical Phase 2] Running {setting} for sub-{args.subject:02d}")
        outputs_by_setting[setting] = run_setting(args, device, setting, conditions)
        print(f"  Averaged outputs: {outputs_by_setting[setting]['averaged_outputs']}")

    manifest_path = write_manifest(args, conditions, outputs_by_setting)
    print(f"[Canonical Phase 2] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
