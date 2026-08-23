"""Frequency-band ablation evaluation for the downloaded NeuroBridge model.

The script keeps the model frozen, removes one EEG band at a time with a
zero-phase Butterworth band-stop filter, and evaluates 200-way image retrieval.
It also measures PSD changes so that the filtering operation can be audited.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch
from torch.utils.data import DataLoader

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.model import EEGProject, TSConv
from module.projector import ProjectorLinear


# Data location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_DIR = SCRIPT_DIR / "data" / "NeuroBridge_download"

# EEG channels used in the original setting
SELECTED_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def checkpoint_expected_channels(checkpoint_path: Path) -> int:
    """Infer the EEG channel count encoded in a NeuroBridge checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    model_state = checkpoint["model_state_dict"]
    if "model.0.weight" in model_state:
        input_dim = int(model_state["model.0.weight"].shape[1])
        if input_dim % 250 != 0:
            raise ValueError(f"Cannot infer channel count from input size {input_dim}.")
        return input_dim // 250
    if "tsconv.4.weight" in model_state:
        return int(model_state["tsconv.4.weight"].shape[2])
    raise ValueError(f"Unsupported EEG encoder in checkpoint: {checkpoint_path}")

# EEG bands may be removed
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# PSD removal script
def remove_frequency_band(
    eeg: np.ndarray,
    low: float, # Lower bound of the band to remove, in Hz
    high: float, # Upper bound of the band to remove, in Hz
    fs: float = 250.0, # Sampling rate of the EEG data, in Hz
    order: int = 4, # Filter order
    pad_seconds: float = 0.5, # Number of seconds to pad around the edges
) -> np.ndarray:
    """Remove ``low``-``high`` Hz along the final (time) axis.

    Reflection padding reduces numerical boundary discontinuities.  Filtering
    is forward-backward (zero phase), and the padded samples are then removed.
    The returned array has exactly the same shape as the input.
    """
    
    eeg = np.asarray(eeg, dtype=np.float32)
    # check whether the band is valid for the given sampling rate
    if not 0.0 < low < high < fs / 2.0:
        raise ValueError(f"Invalid band [{low}, {high}] for fs={fs}")

    # check whether the padding is valid for the given EEG length
    pad_points = int(round(pad_seconds * fs))
    if pad_points < 1 or pad_points >= eeg.shape[-1]:
        raise ValueError(
            f"pad_seconds gives {pad_points} points; expected 1..{eeg.shape[-1] - 1}"
        )

    # design the filter
    sos = butter(
        N=order, # filter order to stablize the filter?
        Wn=[low, high], # cutoff frequencies
        btype="bandstop", # band-stop filter
        fs=fs, # sampling rate of the EEG data, in Hz
        output="sos", # return the filter in SOS form（SOS has stabler output?）
    )
    
    # padding the EEG data to reduce boundary effects
    pad_width = [(0, 0)] * eeg.ndim
    pad_width[-1] = (pad_points, pad_points)
    padded = np.pad(eeg, pad_width=pad_width, mode="reflect")
    
    # filter the EEG data
    filtered = sosfiltfilt(sos, padded, axis=-1, padtype=None)
    result = filtered[..., pad_points:-pad_points].astype(np.float32, copy=False)

    if result.shape != eeg.shape or not np.isfinite(result).all():
        raise RuntimeError("Filtering produced an invalid result")
    return result


# Erase individual differences in PSD by averaging across all examples
def mean_psd(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD across every example and channel."""
    flat = np.asarray(eeg).reshape(-1, eeg.shape[-1])
    frequencies, psd = welch(
        flat,
        fs=fs,
        axis=-1,
        nperseg=flat.shape[-1],
        detrend="constant",
        scaling="density",
    )
    return frequencies, psd.mean(axis=0)

# Compute PSD band power
def band_power(frequencies: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    low, high = band
    keep = (frequencies >= low) & (frequencies <= high)
    if keep.sum() < 2:
        return float("nan")
    return float(trapezoid(psd[keep], frequencies[keep]))

# Compute image retrieval metrics
def retrieval_metrics(
    eeg_features: torch.Tensor,
    image_features: torch.Tensor,
    correct_columns: torch.Tensor,
) -> dict[str, float]:
    """Compute image retrieval using cosine similarity."""
    eeg_features = F.normalize(eeg_features, p=2, dim=1)
    image_features = F.normalize(image_features, p=2, dim=1)
    similarities = eeg_features @ image_features.T
    ranking = similarities.argsort(dim=1, descending=True)
    ranks = (ranking == correct_columns[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1": float((ranks <= 1).float().mean().item()),
        "top5": float((ranks <= 5).float().mean().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "median_rank": float(ranks.float().median().item()),
    }

# Settings of EEGPreImageDataset (the same with sub-08)
def build_dataset(args: argparse.Namespace) -> EEGPreImageDataset:
    return EEGPreImageDataset(
        subject_ids=[args.subject],
        eeg_data_dir=str(args.eeg_data_dir),
        selected_channels=SELECTED_CHANNELS,
        time_window=[0, 250],
        image_feature_dir=str(args.image_feature_dir),
        text_feature_dir="",
        image_aug=True,
        aug_image_feature_dirs=["data/NeuroBridge_download/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"],
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
) -> tuple[torch.nn.Module, ProjectorLinear, ProjectorLinear]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {
        "model_state_dict",
        "eeg_projector_state_dict",
        "img_projector_state_dict",
    }
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        available_keys = list(checkpoint) if isinstance(checkpoint, dict) else []
        missing_keys = sorted(required_keys.difference(available_keys))
        raise ValueError(
            f"Incompatible checkpoint: {checkpoint_path}. Missing keys: "
            f"{missing_keys}; available keys: {available_keys}. This evaluator "
            "requires a NeuroBridge training checkpoint with separate encoder, "
            "EEG-projector, and image-projector state dictionaries."
        )

    # Infer the shared embedding dimension from the saved linear projector.
    projector_weight = checkpoint["eeg_projector_state_dict"]["linear.weight"]
    output_feature_dim, saved_input_dim = projector_weight.shape
    if saved_input_dim != input_feature_dim:
        raise ValueError(
            f"Checkpoint expects image/EEG features of size {saved_input_dim}, "
            f"but the selected image features have size {input_feature_dim}."
        )
        
    model_state = checkpoint["model_state_dict"]
    if "model.0.weight" in model_state:
        encoder_class = EEGProject
    elif "tsconv.0.weight" in model_state and "proj_eeg.0.weight" in model_state:
        encoder_class = TSConv
    else:
        sample_keys = list(model_state)[:10]
        raise ValueError(
            f"Unsupported EEG encoder in checkpoint: {checkpoint_path}. "
            f"Sample model keys: {sample_keys}"
        )

    # Both supported encoders produce input_feature_dim features, followed by
    # the separately saved EEG projector.
    if encoder_class is EEGProject:
        channels_num = int(model_state["model.0.weight"].shape[1]) // 250
    else:
        channels_num = int(model_state["tsconv.4.weight"].shape[2])
    model = encoder_class(
        feature_dim=input_feature_dim,
        eeg_sample_points=250,
        channels_num=channels_num,
    ).to(device)
    eeg_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)
    image_projector = ProjectorLinear(input_feature_dim, output_feature_dim).to(device)

    model.load_state_dict(model_state)
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    image_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    for network in (model, eeg_projector, image_projector):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    return model, eeg_projector, image_projector


def encode_eeg(
    eeg: np.ndarray,
    model: torch.nn.Module,
    projector: ProjectorLinear,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start:start + batch_size]).to(device)
            outputs.append(projector(model(batch)).cpu())
    return torch.cat(outputs, dim=0)


def parse_bands(values: list[list[str]] | None) -> dict[str, tuple[float, float]]:
    if not values:
        return DEFAULT_BANDS.copy()
    bands = {}
    for name, low, high in values:
        bands[name] = (float(low), float(high))
    return bands


def normalize_reference_metric(value: float | None) -> float | None:
    """Accept either a fraction (0.35) or a percentage (35.0)."""
    if value is None:
        return None
    if value < 0:
        raise ValueError("Reference metrics must be non-negative.")
    return value / 100.0 if value > 1.0 else value


def save_psd_plot(
    output_path: Path,
    frequencies: np.ndarray,
    psds: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, psd in psds.items():
        ax.semilogy(frequencies, psd, label=name, linewidth=1.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("EEG PSD before and after band ablation")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR / "intra-subjects_sub-08_checkpoint_last.pth",
    )
    parser.add_argument(
        "--eeg-data-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR / "things_eeg" / "preprocessed_eeg",
    )
    parser.add_argument(
        "--image-feature-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR / "things_eeg" / "image_feature" / "RN50",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "band_ablation_sub08")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--pad-seconds", type=float, default=0.5)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the unmodified preprocessed-EEG reproduction test; do not filter EEG.",
    )
    parser.add_argument(
        "--reference-top1",
        type=float,
        default=None,
        help="Optional reference Top-1, as 0-1 fraction or percentage.",
    )
    parser.add_argument(
        "--reference-top5",
        type=float,
        default=None,
        help="Optional reference Top-5, as 0-1 fraction or percentage.",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=0.005,
        help="Allowed absolute difference from a reference metric (default: 0.005).",
    )
    parser.add_argument(
        "--band",
        nargs=3,
        action="append",
        metavar=("NAME", "LOW", "HIGH"),
        help="Band to remove; repeat this option. Defaults to delta/theta/alpha/beta/gamma.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.subject != 8 and "sub-08" in args.checkpoint.name:
        raise ValueError("The downloaded checkpoint is intra-subject sub-08; use --subject 8.")

    bands = {} if args.baseline_only else parse_bands(args.band)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print("Loading test EEG (the source test.npy is about 1 GB)...")
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    original_eeg = batch[0].numpy().astype(np.float32, copy=False)
    raw_image_features = batch[1].to(device)
    object_indices = batch[4].numpy()
    image_indices = batch[5].numpy()
    print(f"EEG input shape: {original_eeg.shape}")

    model, eeg_projector, image_projector = load_models(
        args.checkpoint,
        input_feature_dim=raw_image_features.shape[-1],
        device=device,
    )
    with torch.inference_mode():
        projected_images = image_projector(raw_image_features).cpu()

    # Map each EEG label to its correct column in the image gallery.
    gallery_keys = list(zip(object_indices.tolist(), image_indices.tolist()))
    key_to_column = {key: column for column, key in enumerate(gallery_keys)}
    if len(key_to_column) != len(gallery_keys):
        raise ValueError("The test image gallery contains duplicate object/image keys.")
    correct_columns = torch.tensor(
        [key_to_column[key] for key in gallery_keys], dtype=torch.long
    )

    results = []

    print("Running original baseline...")
    baseline_features = encode_eeg(
        original_eeg, model, eeg_projector, device, args.batch_size
    )
    baseline = retrieval_metrics(baseline_features, projected_images, correct_columns)
    baseline.update({"condition": "original", "removed_low_hz": None, "removed_high_hz": None})
    results.append(baseline)
    print(f"  Top-1={baseline['top1']:.4f}, Top-5={baseline['top5']:.4f}")

    reference_top1 = normalize_reference_metric(args.reference_top1)
    reference_top5 = normalize_reference_metric(args.reference_top5)
    reference_comparison = {}
    for metric_name, reference_value in (
        ("top1", reference_top1),
        ("top5", reference_top5),
    ):
        if reference_value is not None:
            difference = baseline[metric_name] - reference_value
            reference_comparison[metric_name] = {
                "observed": baseline[metric_name],
                "reference": reference_value,
                "absolute_difference": abs(difference),
                "signed_difference": difference,
                "within_tolerance": abs(difference) <= args.reference_tolerance,
            }
            status = "PASS" if abs(difference) <= args.reference_tolerance else "DIFFERS"
            print(
                f"  {metric_name.upper()} reference={reference_value:.4f}, "
                f"absolute difference={abs(difference):.4f} [{status}]"
            )

    model_configuration = {
        "encoder": "EEGProject",
        "eeg_channels": len(SELECTED_CHANNELS),
        "eeg_sample_points": original_eeg.shape[-1],
        "encoder_input_dim": int(model.model[0].in_features),
        "encoder_output_dim": int(model.model[0].out_features),
        "eeg_projector": "ProjectorLinear",
        "eeg_projector_shape": list(eeg_projector.linear.weight.shape),
        "image_projector": "ProjectorLinear",
        "image_projector_shape": list(image_projector.linear.weight.shape),
        "frozen": True,
        "evaluation_mode": True,
    }
    dataset_configuration = {
        "subject_ids": [args.subject],
        "selected_channels": SELECTED_CHANNELS,
        "time_window": [0, 250],
        "average_repetitions": True,
        "random_subject_or_repetition": False,
        "train": False,
        "eeg_transform": None,
        "eeg_test_aug": False,
        "image_aug": False,
        "image_test_aug": False,
        "number_of_test_examples": len(dataset),
        "eeg_shape_after_loading": list(original_eeg.shape),
        "raw_image_feature_dim": int(raw_image_features.shape[-1]),
    }

    if args.baseline_only:
        validation_path = args.output_dir / "baseline_reproduction.json"
        validation_payload = {
            "purpose": "Unmodified preprocessed-EEG checkpoint reproduction test",
            "filter_applied": False,
            "subject": args.subject,
            "checkpoint": str(args.checkpoint.resolve()),
            "eeg_data_dir": str(args.eeg_data_dir.resolve()),
            "image_feature_dir": str(args.image_feature_dir.resolve()),
            "dataset_configuration": dataset_configuration,
            "model_configuration": model_configuration,
            "observed_metrics": baseline,
            "reference_tolerance": args.reference_tolerance,
            "reference_comparison": reference_comparison,
        }
        with validation_path.open("w", encoding="utf-8") as file:
            json.dump(validation_payload, file, ensure_ascii=False, indent=2)
        print("Baseline-only mode: no frequency filter was designed or applied.")
        print(f"Saved: {validation_path}")
        return

    frequencies, original_psd = mean_psd(original_eeg, args.fs)
    original_band_powers = {
        name: band_power(frequencies, original_psd, band) for name, band in bands.items()
    }
    psds = {"original": original_psd}
    detailed_power_ratios = {}
    for condition, (low, high) in bands.items():
        print(f"Removing {condition}: {low:g}-{high:g} Hz...")
        ablated_eeg = remove_frequency_band(
            original_eeg,
            low=low,
            high=high,
            fs=args.fs,
            order=args.filter_order,
            pad_seconds=args.pad_seconds,
        )
        eeg_features = encode_eeg(
            ablated_eeg, model, eeg_projector, device, args.batch_size
        )
        metrics = retrieval_metrics(eeg_features, projected_images, correct_columns)
        _, filtered_psd = mean_psd(ablated_eeg, args.fs)
        psds[condition] = filtered_psd

        ratios = {}
        for measured_name, measured_band in bands.items():
            filtered_power = band_power(frequencies, filtered_psd, measured_band)
            denominator = original_band_powers[measured_name]
            ratios[measured_name] = filtered_power / denominator if denominator > 0 else float("nan")
        detailed_power_ratios[condition] = ratios

        metrics.update({
            "condition": condition,
            "removed_low_hz": low,
            "removed_high_hz": high,
            "top1_drop": baseline["top1"] - metrics["top1"],
            "top5_drop": baseline["top5"] - metrics["top5"],
            "target_band_power_ratio": ratios[condition],
        })
        results.append(metrics)
        print(
            f"  Top-1={metrics['top1']:.4f}, Top-5={metrics['top5']:.4f}, "
            f"target PSD ratio={ratios[condition]:.4f}"
        )

    csv_path = args.output_dir / "retrieval_results.csv"
    fieldnames = [
        "condition", "removed_low_hz", "removed_high_hz", "top1", "top5",
        "mean_rank", "median_rank", "top1_drop", "top5_drop", "target_band_power_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    json_path = args.output_dir / "ablation_details.json"
    payload = {
        "subject": args.subject,
        "checkpoint": str(args.checkpoint.resolve()),
        "eeg_shape": list(original_eeg.shape),
        "selected_channels": SELECTED_CHANNELS,
        "dataset_configuration": dataset_configuration,
        "model_configuration": model_configuration,
        "sampling_rate": args.fs,
        "filter_order": args.filter_order,
        "pad_seconds": args.pad_seconds,
        "bands": bands,
        "results": results,
        "reference_comparison": reference_comparison,
        "band_power_ratios": detailed_power_ratios,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    plot_path = args.output_dir / "psd_comparison.png"
    save_psd_plot(plot_path, frequencies, psds)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
