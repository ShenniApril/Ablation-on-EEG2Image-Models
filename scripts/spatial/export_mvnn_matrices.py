"""Export the fixed, session-wise MVNN matrices used by NeuroBridge.

The matrices are estimated from the unmodified TRAINING epochs of each
session.  They can then be reused for clean and microstate-ablated test EEG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from sklearn.discriminant_analysis import _cov
from tqdm import tqdm

from preprocess_eeg import preprocess


def estimate_mvnn_matrix(epoched_train: np.ndarray) -> np.ndarray:
    """Match ``preprocess_eeg.mvnn`` and return Sigma_train ** -0.5."""
    if epoched_train.ndim != 4:
        raise ValueError(
            "Expected session training EEG shaped "
            f"(conditions, repetitions, channels, time), got {epoched_train.shape}."
        )

    condition_covariances = np.empty(
        (epoched_train.shape[0], epoched_train.shape[2], epoched_train.shape[2]),
        dtype=np.float64,
    )
    for condition in tqdm(
        range(epoched_train.shape[0]), desc="Training-condition covariances"
    ):
        condition_data = epoched_train[condition]
        condition_covariances[condition] = np.mean(
            [
                _cov(epoch.T, shrinkage="auto")
                for epoch in condition_data
            ],
            axis=0,
        )

    sigma_train = condition_covariances.mean(axis=0)
    matrix = scipy.linalg.fractional_matrix_power(sigma_train, -0.5)
    matrix = np.real_if_close(matrix, tol=1000)
    if np.iscomplexobj(matrix):
        raise RuntimeError("MVNN inverse square root contains complex values.")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("MVNN matrix contains non-finite values.")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-dir", type=Path, default=Path("data/things_eeg/raw_eeg")
    )
    parser.add_argument(
        "--pre-mvnn-info",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn/info.json"),
        help="Provides the exact 63-channel order used for the no-MVNN arrays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mvnn_matrices_sub08"),
    )
    parser.add_argument("--subject", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--rfreq", type=int, default=250)
    parser.add_argument("--baseline-duration", type=float, default=0.2)
    parser.add_argument("--after-duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20200220)
    args = parser.parse_args()

    with args.pre_mvnn_info.open("r", encoding="utf-8") as file:
        info = json.load(file)
    channels_order = info["ch_names"]
    if len(channels_order) != 63:
        raise ValueError(f"Expected 63 channels, found {len(channels_order)}.")

    # ``preprocess`` only reads these attributes.
    preprocess_args = argparse.Namespace(
        baseline_duration=args.baseline_duration,
        after_duration=args.after_duration,
        rfreq=args.rfreq,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "subject": args.subject,
        "sessions": args.sessions,
        "source": "unmodified training epochs",
        "estimator": "mean Ledoit-Wolf covariance per epoch/condition",
        "channel_order": channels_order,
        "rfreq": args.rfreq,
        "baseline_duration": args.baseline_duration,
        "after_duration": args.after_duration,
        "seed": args.seed,
        "matrices": [],
    }

    for session in range(1, args.sessions + 1):
        train_path = (
            args.raw_data_dir
            / f"sub-{args.subject:02d}"
            / f"ses-{session:02d}"
            / "raw_eeg_training.npy"
        )
        if not train_path.exists():
            raise FileNotFoundError(train_path)
        print(f"\nPreparing training epochs for session {session:02d}: {train_path}")
        train_data, _, returned_channels, _, sampling_rate = preprocess(
            str(train_path), "train", channels_order, preprocess_args, args.seed
        )
        if returned_channels != channels_order:
            raise RuntimeError("Returned channel order differs from pre-MVNN data.")

        matrix = estimate_mvnn_matrix(train_data)
        output_path = args.output_dir / f"ses-{session:02d}_mvnn.npy"
        np.save(output_path, matrix)
        metadata["matrices"].append(
            {
                "session": session,
                "path": output_path.name,
                "shape": list(matrix.shape),
                "sampling_rate": float(sampling_rate),
            }
        )
        print(f"Saved {output_path} {matrix.shape}")

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"\nSaved all fixed MVNN matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
