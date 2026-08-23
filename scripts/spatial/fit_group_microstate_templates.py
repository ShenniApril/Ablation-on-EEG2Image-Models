"""Fit one balanced group-level microstate template set from training EEG.

Each subject contributes the same requested number of training trials. GFP
peak maps are extracted subject-by-subject and then pooled for clustering.
Only training EEG is accepted by convention; test EEG must never be supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microstate_clustering_reference import (
    collect_gfp_peak_maps,
    fit_polarity_invariant_microstates,
    flatten_eeg_file,
)


def subject_path(root: Path, subject: int) -> Path:
    return root / f"sub-{subject:02d}" / "train.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eeg-root",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--trials-per-subject", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--max-peaks-per-trial", type=int, default=12)
    parser.add_argument("--min-peak-distance-ms", type=float, default=10.0)
    parser.add_argument("--gfp-percentile", type=float, default=None)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=2025)
    args = parser.parse_args()

    if args.trials_per_subject <= 0:
        raise ValueError("--trials-per-subject must be positive.")
    if len(set(args.subjects)) != len(args.subjects):
        raise ValueError("--subjects contains duplicates.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)
    all_maps: list[np.ndarray] = []
    all_gfp: list[np.ndarray] = []
    subject_diagnostics: list[dict[str, object]] = []
    expected_epoch_shape: tuple[int, int] | None = None

    for subject in args.subjects:
        path = subject_path(args.eeg_root, subject)
        if not path.is_file():
            raise FileNotFoundError(f"Missing training EEG for sub-{subject:02d}: {path}")
        array = np.load(path, mmap_mode="r")
        trials, leading_shape = flatten_eeg_file(array)
        epoch_shape = (int(trials.shape[1]), int(trials.shape[2]))
        if expected_epoch_shape is None:
            expected_epoch_shape = epoch_shape
        elif epoch_shape != expected_epoch_shape:
            raise ValueError(
                f"sub-{subject:02d} epoch shape {epoch_shape} differs from "
                f"the expected {expected_epoch_shape}."
            )
        if len(trials) < args.trials_per_subject:
            raise ValueError(
                f"sub-{subject:02d} has {len(trials)} trials, fewer than "
                f"--trials-per-subject={args.trials_per_subject}."
            )

        indices = np.sort(
            rng.choice(len(trials), size=args.trials_per_subject, replace=False)
        )
        sampled = np.asarray(trials[indices])
        maps, _, peak_gfp = collect_gfp_peak_maps(
            sampled,
            sampling_rate=args.sampling_rate,
            min_peak_distance_ms=args.min_peak_distance_ms,
            max_peaks_per_trial=args.max_peaks_per_trial,
            gfp_percentile=args.gfp_percentile,
        )
        all_maps.append(maps)
        all_gfp.append(peak_gfp)
        subject_diagnostics.append(
            {
                "subject": f"sub-{subject:02d}",
                "eeg_path": str(path.resolve()),
                "array_shape": list(array.shape),
                "leading_shape": list(leading_shape),
                "available_trials": int(len(trials)),
                "sampled_trials": int(len(indices)),
                "sampled_trial_indices": indices.tolist(),
                "number_of_peak_maps": int(len(maps)),
            }
        )
        print(
            f"sub-{subject:02d}: sampled {len(indices)}/{len(trials)} trials, "
            f"collected {len(maps)} GFP peak maps"
        )

    pooled_maps = np.concatenate(all_maps, axis=0)
    pooled_gfp = np.concatenate(all_gfp, axis=0)
    templates, clustering = fit_polarity_invariant_microstates(
        pooled_maps,
        peak_gfp=pooled_gfp,
        n_states=args.n_states,
        n_init=args.n_init,
        random_state=args.random_state,
    )
    template_path = args.output_dir / "microstate_templates.npy"
    np.save(template_path, templates)

    metadata = {
        "template_scope": "balanced_group_training_only",
        "subjects": [f"sub-{subject:02d}" for subject in args.subjects],
        "eeg_root": str(args.eeg_root.resolve()),
        "n_states": args.n_states,
        "sampling_rate": args.sampling_rate,
        "trials_per_subject": args.trials_per_subject,
        "total_sampled_trials": args.trials_per_subject * len(args.subjects),
        "max_peaks_per_trial": args.max_peaks_per_trial,
        "min_peak_distance_ms": args.min_peak_distance_ms,
        "gfp_percentile": args.gfp_percentile,
        "n_init": args.n_init,
        "random_state": args.random_state,
        "epoch_shape_channels_time": list(expected_epoch_shape or ()),
        "clustering_diagnostics": clustering,
        "subject_diagnostics": subject_diagnostics,
        "leakage_policy": "Templates were fitted from each subject's train.npy only.",
    }
    with (args.output_dir / "group_microstate_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Templates: {templates.shape}")
    print(f"Total peak maps: {len(pooled_maps)}")
    print(f"Saved: {template_path}")


if __name__ == "__main__":
    main()
