"""Batch-run and summarize microstate clustering parameter experiments.

Example
-------
Pilot comparison around the current K=4 configuration:

python batch_microstate_clustering.py `
  --n-states 4 `
  --fit-trials 5000 10000 `
  --max-peaks 10 12 `
  --peak-distance-ms 10 20 `
  --min-segment-ms 10 20 `
  --n-init 50 `
  --seeds 2025 `
  --max-runs 16

The script creates one directory per configuration and writes
``clustering_experiment_summary.csv`` in the experiment root.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def configuration_name(config: dict[str, int | float]) -> str:
    """Create a readable, collision-resistant directory name."""
    return (
        f"k{config['n_states']}"
        f"_trials{config['fit_trials']}"
        f"_peaks{config['max_peaks']}"
        f"_distance{config['peak_distance_ms']:g}ms"
        f"_segment{config['min_segment_ms']:g}ms"
        f"_init{config['n_init']}"
        f"_seed{config['seed']}"
    )


def template_redundancy(templates: np.ndarray) -> tuple[float, float]:
    """Return maximum and mean off-diagonal absolute template correlation."""
    templates = np.asarray(templates, dtype=np.float64)
    templates -= templates.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates /= np.maximum(norms, 1e-12)
    correlations = np.abs(templates @ templates.T)
    off_diagonal = correlations[~np.eye(len(templates), dtype=bool)]
    if not len(off_diagonal):
        return 0.0, 0.0
    return float(off_diagonal.max()), float(off_diagonal.mean())


def summarize_run(run_dir: Path) -> dict[str, object]:
    """Read saved clustering outputs and calculate selection diagnostics."""
    with (run_dir / "microstate_metadata.json").open(
        "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    n_states = int(metadata["n_states"])
    split = metadata["split_diagnostics"]["test"]
    feature_file = np.load(run_dir / "test_microstate_features.npz")
    coverage = np.asarray(feature_file["coverage"]).reshape(-1, n_states)
    duration = np.asarray(feature_file["mean_duration_ms"]).reshape(
        -1, n_states
    )
    occurrence = np.asarray(feature_file["occurrence_hz"]).reshape(
        -1, n_states
    )
    templates = np.load(run_dir / "microstate_templates.npy")
    max_corr, mean_corr = template_redundancy(templates)

    mean_coverage = coverage.mean(axis=0)
    median_coverage = np.median(coverage, axis=0)
    trial_presence = (coverage > 0).mean(axis=0)
    mean_duration = duration.mean(axis=0)
    mean_occurrence = occurrence.mean(axis=0)

    train = metadata["split_diagnostics"].get("train", {})
    clustering = metadata["clustering_diagnostics"]
    row: dict[str, object] = {
        "run": run_dir.name,
        "status": "complete",
        "n_states": n_states,
        "fit_trials": metadata["max_fit_trials"],
        "max_peaks": metadata["max_peaks_per_trial"],
        "peak_distance_ms": metadata["min_peak_distance_ms"],
        "min_segment_ms": metadata["min_segment_ms"],
        "n_init": metadata["n_init"],
        "seed": metadata["random_state"],
        "number_of_peak_maps": clustering["number_of_peak_maps"],
        "peak_map_gev": clustering[
            "peak_map_global_explained_variance"
        ],
        "test_mean_abs_r": split[
            "mean_backfit_absolute_spatial_correlation"
        ],
        "test_gev": split["global_explained_variance"],
        "train_mean_abs_r": train.get(
            "mean_backfit_absolute_spatial_correlation", ""
        ),
        "train_gev": train.get("global_explained_variance", ""),
        "min_mean_coverage": float(mean_coverage.min()),
        "min_trial_presence": float(trial_presence.min()),
        "max_template_abs_correlation": max_corr,
        "mean_template_abs_correlation": mean_corr,
        "coverage_warning": int(
            mean_coverage.min() < 0.02 or trial_presence.min() < 0.20
        ),
        "redundancy_warning": int(max_corr > 0.90),
    }
    for state in range(n_states):
        display = state + 1
        row[f"state{display}_mean_coverage"] = float(mean_coverage[state])
        row[f"state{display}_median_coverage"] = float(
            median_coverage[state]
        )
        row[f"state{display}_trial_presence"] = float(trial_presence[state])
        row[f"state{display}_mean_duration_ms"] = float(
            mean_duration[state]
        )
        row[f"state{display}_mean_occurrence_hz"] = float(
            mean_occurrence[state]
        )
    return row


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    """Write a rectangular CSV even when runs have different K values."""
    preferred = [
        "run",
        "status",
        "error",
        "n_states",
        "fit_trials",
        "max_peaks",
        "peak_distance_ms",
        "min_segment_ms",
        "n_init",
        "seed",
        "number_of_peak_maps",
        "peak_map_gev",
        "test_mean_abs_r",
        "test_gev",
        "train_mean_abs_r",
        "train_gev",
        "min_mean_coverage",
        "min_trial_presence",
        "max_template_abs_correlation",
        "mean_template_abs_correlation",
        "coverage_warning",
        "redundancy_warning",
    ]
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    fields = [name for name in preferred if name in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-eeg",
        type=Path,
        default=Path(
            "data/things_eeg/preprocessed_eeg_no_mvnn/sub-08/train.npy"
        ),
    )
    parser.add_argument(
        "--apply-eeg",
        type=Path,
        default=Path(
            "data/things_eeg/preprocessed_eeg_no_mvnn/sub-08/test.npy"
        ),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("results/microstate_parameter_search"),
    )
    parser.add_argument("--n-states", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--fit-trials", type=int, nargs="+", default=[10000]
    )
    parser.add_argument("--max-peaks", type=int, nargs="+", default=[12])
    parser.add_argument(
        "--peak-distance-ms", type=float, nargs="+", default=[20.0]
    )
    parser.add_argument(
        "--min-segment-ms", type=float, nargs="+", default=[20.0]
    )
    parser.add_argument("--n-init", type=int, nargs="+", default=[50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--backfit-batch-size", type=int, default=256)
    parser.add_argument(
        "--backfit-train",
        action="store_true",
        help="Also save complete-train labels/features; substantially slower.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=32,
        help="Safety limit for the Cartesian parameter grid.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run configurations again even if metadata already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running clustering.",
    )
    args = parser.parse_args()

    grid_keys = [
        "n_states",
        "fit_trials",
        "max_peaks",
        "peak_distance_ms",
        "min_segment_ms",
        "n_init",
        "seed",
    ]
    grid_values = [
        args.n_states,
        args.fit_trials,
        args.max_peaks,
        args.peak_distance_ms,
        args.min_segment_ms,
        args.n_init,
        args.seeds,
    ]
    configurations = [
        dict(zip(grid_keys, values))
        for values in itertools.product(*grid_values)
    ]
    if len(configurations) > args.max_runs:
        raise ValueError(
            f"The requested grid contains {len(configurations)} runs, exceeding "
            f"--max-runs {args.max_runs}. Reduce the grid or explicitly increase "
            "the safety limit."
        )

    script = Path(__file__).with_name("microstate_clustering_reference.py")
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    print(f"Planned configurations: {len(configurations)}")
    print(f"Experiment root: {args.experiment_root.resolve()}")
    rows: list[dict[str, object]] = []

    for index, config in enumerate(configurations, start=1):
        run_name = configuration_name(config)
        run_dir = args.experiment_root / run_name
        metadata_path = run_dir / "microstate_metadata.json"
        command = [
            sys.executable,
            str(script),
            "--fit-eeg",
            str(args.fit_eeg),
            "--apply-eeg",
            str(args.apply_eeg),
            "--output-dir",
            str(run_dir),
            "--n-states",
            str(config["n_states"]),
            "--max-fit-trials",
            str(config["fit_trials"]),
            "--max-peaks-per-trial",
            str(config["max_peaks"]),
            "--min-peak-distance-ms",
            str(config["peak_distance_ms"]),
            "--min-segment-ms",
            str(config["min_segment_ms"]),
            "--n-init",
            str(config["n_init"]),
            "--backfit-batch-size",
            str(args.backfit_batch_size),
            "--sampling-rate",
            str(args.sampling_rate),
            "--random-state",
            str(config["seed"]),
        ]
        if args.backfit_train:
            command.append("--backfit-train")

        print(f"\n[{index}/{len(configurations)}] {run_name}")
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue

        if metadata_path.exists() and not args.rerun:
            print("Already complete; summarizing existing outputs.")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                print(f"FAILED; inspect {log_path}")
                rows.append(
                    {
                        "run": run_name,
                        "status": "failed",
                        "error": f"exit_code={result.returncode}",
                        **config,
                    }
                )
                write_summary(
                    rows,
                    args.experiment_root
                    / "clustering_experiment_summary.csv",
                )
                continue

        try:
            rows.append(summarize_run(run_dir))
        except Exception as error:  # Keep the remaining grid running.
            print(f"SUMMARY FAILED: {error}")
            rows.append(
                {
                    "run": run_name,
                    "status": "summary_failed",
                    "error": repr(error),
                    **config,
                }
            )
        write_summary(
            rows,
            args.experiment_root / "clustering_experiment_summary.csv",
        )

    if args.dry_run:
        print("\nDry run complete; no clustering was executed.")
    else:
        summary_path = (
            args.experiment_root / "clustering_experiment_summary.csv"
        )
        print(f"\nSaved summary: {summary_path}")
        print(
            "Sort/filter the CSV using test_gev, min_mean_coverage, "
            "min_trial_presence and max_template_abs_correlation together."
        )


if __name__ == "__main__":
    main()
