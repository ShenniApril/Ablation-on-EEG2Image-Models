"""STATE discovery/back-fitting pipeline for THINGS-EEG.

Recommended protocol
--------------------
1. Fit one subject-specific microstate codebook from TRAINING trials only.
2. Pool GFP-peak scalp maps across trials/images for clustering, but never
   treat the end of one trial and the start of the next as temporally adjacent.
3. Freeze the templates and back-fit every test trial independently.
4. Compute duration/occurrence/coverage/transition features within each trial,
   so transitions across trial boundaries are never counted.

This is a compact research reference, not a replacement for a validated
microstate package such as PyCrostates.  For physiological topography claims,
prefer full-channel, single-trial EEG before MVNN spatial whitening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# mainly retrain the features relecting the spatial topographical info
def normalize_topographies(maps: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Average-reference and L2-normalize scalp maps.

    Parameters
    ----------
    maps
        Array shaped (..., channels).
    """
    maps = np.asarray(maps, dtype=np.float64)
    maps = maps - maps.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(maps, axis=-1, keepdims=True)
    return maps / np.maximum(norm, eps)

# determine a certain direction for microstate topography to ensure we find the template with the maximum regardless of the direction
def canonicalize_polarity(maps: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign for display; clustering remains sign-free."""
    maps = np.asarray(maps).copy()
    largest = np.argmax(np.abs(maps), axis=1)
    signs = np.sign(maps[np.arange(len(maps)), largest])
    signs[signs == 0] = 1
    return maps * signs[:, None]


def collect_gfp_peak_maps(
    trials: np.ndarray,
    sampling_rate: float = 250.0, # same with the EEG sampling rate
    min_peak_distance_ms: float = 20.0,# to ensure the peak is not too close to the previous one
    max_peaks_per_trial: int = 12, # as most microstate lasts for 80-120ms, the whole EEG signal lasts for 1000ms, so we can have at most 12 peaks
    gfp_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool GFP-peak maps while retaining the source trial of every map.

    ``trials`` must be shaped (trials, channels, time).  The function pools
    topographies, not raw time series, so concatenation boundaries cannot
    create artificial transitions.
    """
    trials = np.asarray(trials)
    if trials.ndim != 3:
        raise ValueError(f"Expected (trials, channels, time), got {trials.shape}")

    min_distance = max(1, int(round(min_peak_distance_ms * sampling_rate / 1000)))
    pooled_maps: list[np.ndarray] = []
    source_trials: list[np.ndarray] = []
    peak_gfp_values: list[np.ndarray] = []

    for trial_index, trial in enumerate(trials):
        # GFP is the across-channel standard deviation at every time point.
        gfp = np.std(trial, axis=0)
        peaks, _ = find_peaks(gfp, distance=min_distance)
        if len(peaks) == 0:
            continue

        if gfp_percentile is not None:
            threshold = np.percentile(gfp, gfp_percentile)
            peaks = peaks[gfp[peaks] >= threshold]
        if len(peaks) == 0:
            continue

        # Keep the strongest peaks to prevent a few trials dominating the pool.
        order = np.argsort(gfp[peaks])[::-1]
        peaks = peaks[order[:max_peaks_per_trial]]
        pooled_maps.append(trial[:, peaks].T)
        source_trials.append(np.full(len(peaks), trial_index, dtype=np.int64))
        peak_gfp_values.append(gfp[peaks])

    if not pooled_maps:
        raise ValueError("No GFP peaks were found.")

    maps = normalize_topographies(np.concatenate(pooled_maps, axis=0))
    sources = np.concatenate(source_trials, axis=0)
    peak_gfp = np.concatenate(peak_gfp_values).astype(np.float64)
    return maps, sources, peak_gfp


def fit_polarity_invariant_microstates(
    peak_maps: np.ndarray,
    peak_gfp: np.ndarray | None = None,
    n_states: int = 4, # number of microstate to cluster
    n_init: int = 20, # number of random initialization
    max_iter: int = 200, # maximum number of iterations
    random_state: int = 2025,
) -> tuple[np.ndarray, dict[str, float]]:
    """Modified k-means using absolute spatial correlation.

    Assignment ignores map polarity.  Each cluster template is updated with
    the principal eigenvector of its topographic covariance matrix.
    """
    x = normalize_topographies(peak_maps)
    if len(x) < n_states:
        raise ValueError("Fewer peak maps than requested microstate classes.")

    rng = np.random.default_rng(random_state)
    if peak_gfp is None:
        weights = np.ones(len(x), dtype=np.float64)
    else:
        weights = np.asarray(peak_gfp, dtype=np.float64) ** 2
        if weights.shape != (len(x),):
            raise ValueError("peak_gfp must contain one value per peak map.")
    best_score = -np.inf
    best_templates = None
    best_iterations = 0

    for _ in range(n_init):
        # Initialize templates picked from peak maps
        templates = x[rng.choice(len(x), size=n_states, replace=False)].copy()
        templates = normalize_topographies(templates)
        previous_labels = None

        for iteration in range(1, max_iter + 1):
            # x and templates are normalized,  dot product equals spatial correlation
            correlations = x @ templates.T
            labels = np.argmax(np.abs(correlations), axis=1)
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels.copy()

            updated = []
            empty_cluster = False
            for state in range(n_states):
                members = x[labels == state]
                if len(members) == 0:
                    empty_cluster = True
                    break
                covariance = members.T @ members
                # principal eigenvectors represent the newly clustered microstates
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                updated.append(eigenvectors[:, np.argmax(eigenvalues)])
            if empty_cluster:
                break
            templates = normalize_topographies(np.stack(updated))

        correlations = x @ templates.T
        explained = np.max(np.abs(correlations), axis=1) ** 2
        # GFP-weighted Global Explained Variance (GEV).
        score = float(np.sum(weights * explained) / np.maximum(weights.sum(), 1e-12))
        if score > best_score:
            best_score = score
            best_templates = templates.copy()
            best_iterations = iteration

    if best_templates is None:
        raise RuntimeError("Microstate clustering failed.")

    best_templates = canonicalize_polarity(normalize_topographies(best_templates))
    diagnostics = {
        "peak_map_global_explained_variance": best_score,
        "iterations_in_best_initialization": int(best_iterations),
        "number_of_peak_maps": int(len(x)),
    }
    return best_templates.astype(np.float32), diagnostics


def smooth_short_segments(
    labels: np.ndarray,
    absolute_correlations: np.ndarray,
    min_segment_samples: int,
    max_passes: int = 20,
) -> np.ndarray:
    """Relabel very short segments using the better adjacent state.

    Smoothing is performed independently within every trial. It cannot create
    a transition across a trial boundary.
    """
    if min_segment_samples <= 1:
        return labels
    smoothed = labels.copy()
    n_states = absolute_correlations.shape[-1]

    for trial_index in range(len(smoothed)):
        sequence = smoothed[trial_index]
        scores = absolute_correlations[trial_index]
        for _ in range(max_passes):
            starts = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]
            ends = np.r_[starts[1:], len(sequence)]
            changed = False
            for segment_index, (start, end) in enumerate(zip(starts, ends)):
                if end - start >= min_segment_samples:
                    continue
                candidates = []
                if segment_index > 0:
                    candidates.append(int(sequence[start - 1]))
                if segment_index + 1 < len(starts):
                    candidates.append(int(sequence[end]))
                candidates = sorted(set(candidates))
                candidates = [state for state in candidates if 0 <= state < n_states]
                if not candidates:
                    continue
                candidate_scores = [scores[start:end, state].mean() for state in candidates]
                sequence[start:end] = candidates[int(np.argmax(candidate_scores))]
                changed = True
            if not changed:
                break
    return smoothed


def backfit_microstates(
    trials: np.ndarray,
    templates: np.ndarray,
    min_segment_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every time point to the template with maximum |correlation|.

    Returns
    -------
    labels
        Integer array shaped (trials, time).
    winning_correlation
        Absolute spatial correlation shaped (trials, time).
    """
    trials = np.asarray(trials)
    if trials.ndim != 3:
        raise ValueError(f"Expected (trials, channels, time), got {trials.shape}")
    if trials.shape[1] != templates.shape[1]:
        raise ValueError("Trial and template channel counts do not match.")

    # (trial, channel, time) -> (trial, time, channel)
    maps = np.moveaxis(trials, 1, 2)
    maps = normalize_topographies(maps)
    correlations = np.einsum("ntc,kc->ntk", maps, normalize_topographies(templates))
    absolute = np.abs(correlations)
    labels = np.argmax(absolute, axis=-1)
    labels = smooth_short_segments(labels, absolute, min_segment_samples)
    winning = np.take_along_axis(absolute, labels[..., None], axis=-1)[..., 0]
    return labels.astype(np.int8), winning.astype(np.float32)


def microstate_features(
    labels: np.ndarray,
    n_states: int,
    sampling_rate: float = 250.0,
    trials: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute trial-wise features without crossing trial boundaries."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected (trials, time), got {labels.shape}")

    n_trials, n_times = labels.shape
    coverage = np.zeros((n_trials, n_states), dtype=np.float32)
    occurrence_hz = np.zeros_like(coverage)
    mean_duration_ms = np.zeros_like(coverage)
    transitions = np.zeros((n_trials, n_states, n_states), dtype=np.float32)
    mean_gfp = np.zeros_like(coverage)
    trial_seconds = n_times / sampling_rate
    if trials is not None:
        trials = np.asarray(trials)
        if trials.shape[0] != n_trials or trials.shape[-1] != n_times:
            raise ValueError("trials and labels do not describe the same EEG epochs.")
        gfp = np.std(trials, axis=1)
    else:
        gfp = None

    for trial_index, sequence in enumerate(labels):
        coverage[trial_index] = np.bincount(sequence, minlength=n_states) / n_times

        starts = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]
        ends = np.r_[starts[1:], n_times]
        states = sequence[starts]
        durations = ends - starts

        for state in range(n_states):
            state_durations = durations[states == state]
            occurrence_hz[trial_index, state] = len(state_durations) / trial_seconds
            if len(state_durations):
                mean_duration_ms[trial_index, state] = (
                    state_durations.mean() * 1000.0 / sampling_rate
                )
            if gfp is not None and np.any(sequence == state):
                mean_gfp[trial_index, state] = gfp[trial_index, sequence == state].mean()

        # Only consecutive segments within this trial contribute transitions.
        for source, target in zip(states[:-1], states[1:]):
            transitions[trial_index, source, target] += 1
        row_sum = transitions[trial_index].sum(axis=1, keepdims=True)
        transitions[trial_index] /= np.maximum(row_sum, 1.0)

    result = {
        "coverage": coverage,
        "occurrence_hz": occurrence_hz,
        "mean_duration_ms": mean_duration_ms,
        "transition_probability": transitions,
    }
    if gfp is not None:
        result["mean_gfp_amplitude"] = mean_gfp
    return result


def global_explained_variance(
    trials: np.ndarray,
    winning_correlation: np.ndarray,
) -> float:
    """Compute GFP-weighted GEV over all time points."""
    gfp_squared = np.std(trials, axis=1, dtype=np.float64) ** 2
    numerator = np.sum(gfp_squared * np.asarray(winning_correlation) ** 2)
    return float(numerator / np.maximum(gfp_squared.sum(), 1e-12))


def flatten_eeg_file(array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten all leading EEG dimensions into trials and retain their shape."""
    if array.ndim < 3:
        raise ValueError(f"EEG array has too few dimensions: {array.shape}")
    leading_shape = tuple(array.shape[:-2])
    trials = array.reshape(-1, array.shape[-2], array.shape[-1])
    return trials, leading_shape


def select_fit_trials(
    trials: np.ndarray,
    max_fit_trials: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample trials for template discovery."""
    if max_fit_trials <= 0 or len(trials) <= max_fit_trials:
        indices = np.arange(len(trials))
    else:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(trials), max_fit_trials, replace=False))
    return np.asarray(trials[indices]), indices


def backfit_and_save_dataset(
    eeg_path: Path,
    output_dir: Path,
    output_prefix: str,
    templates: np.ndarray,
    n_states: int,
    sampling_rate: float,
    min_segment_ms: float,
    batch_size: int,
) -> dict[str, object]:
    """Back-fit one EEG split in bounded-memory batches and save its outputs."""
    print(f"Memory-mapping {output_prefix} back-fit EEG: {eeg_path}")
    eeg_array = np.load(eeg_path, mmap_mode="r")
    trials, leading_shape = flatten_eeg_file(eeg_array)
    min_segment_samples = max(
        1, int(round(min_segment_ms * sampling_rate / 1000.0))
    )

    label_batches = []
    correlation_batches = []
    feature_batches: dict[str, list[np.ndarray]] = {}
    gev_numerator = 0.0
    gev_denominator = 0.0

    for start in range(0, len(trials), batch_size):
        trial_batch = np.asarray(trials[start:start + batch_size])
        labels, winning = backfit_microstates(
            trial_batch,
            templates,
            min_segment_samples=min_segment_samples,
        )
        features = microstate_features(
            labels,
            n_states=n_states,
            sampling_rate=sampling_rate,
            trials=trial_batch,
        )
        label_batches.append(labels)
        correlation_batches.append(winning)
        for name, values in features.items():
            feature_batches.setdefault(name, []).append(values)

        gfp_squared = np.std(trial_batch, axis=1, dtype=np.float64) ** 2
        gev_numerator += float(np.sum(gfp_squared * winning.astype(np.float64) ** 2))
        gev_denominator += float(gfp_squared.sum())

    all_labels = np.concatenate(label_batches, axis=0)
    all_correlations = np.concatenate(correlation_batches, axis=0)
    all_features = {
        name: np.concatenate(values, axis=0)
        for name, values in feature_batches.items()
    }

    np.save(
        output_dir / f"{output_prefix}_microstate_labels.npy",
        all_labels.reshape(*leading_shape, all_labels.shape[-1]),
    )
    np.save(
        output_dir / f"{output_prefix}_winning_spatial_correlation.npy",
        all_correlations.reshape(*leading_shape, all_correlations.shape[-1]),
    )
    np.savez_compressed(
        output_dir / f"{output_prefix}_microstate_features.npz",
        **{
            name: values.reshape(*leading_shape, *values.shape[1:])
            for name, values in all_features.items()
        },
    )

    gev = gev_numerator / max(gev_denominator, 1e-12)
    print(
        f"{output_prefix}: labels={all_labels.shape}, "
        f"mean |r|={all_correlations.mean():.4f}, GEV={gev:.4f}"
    )
    return {
        "eeg_path": str(eeg_path.resolve()),
        "array_shape": list(eeg_array.shape),
        "leading_shape": list(leading_shape),
        "number_of_trials": int(len(trials)),
        "min_segment_ms": min_segment_ms,
        "min_segment_samples": min_segment_samples,
        "mean_backfit_absolute_spatial_correlation": float(all_correlations.mean()),
        "global_explained_variance": float(gev),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-eeg",
        type=Path,
        nargs="+",
        help="One or more training EEG files used to learn common templates.",
    )
    parser.add_argument(
        "--templates-in", type=Path,
        help="Use frozen templates instead of fitting new ones.",
    )
    parser.add_argument(
        "--apply-eeg",
        type=Path,
        required=False,
        help="Test EEG .npy to back-fit with the frozen training templates.",
    )
    parser.add_argument(
        "--backfit-train",
        action="store_true",
        help="Also back-fit the complete training file and save train features/sequences.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--max-fit-trials", type=int, default=5000)
    parser.add_argument("--max-peaks-per-trial", type=int, default=12)
    parser.add_argument("--min-peak-distance-ms", type=float, default=20.0)
    parser.add_argument("--gfp-percentile", type=float, default=None)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument(
        "--min-segment-ms",
        type=float,
        default=20.0,
        help="Relabel fitted segments shorter than this duration; use 0 to disable.",
    )
    parser.add_argument("--backfit-batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=2025)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_sources = []
    sampled_trials = np.empty((0, 0, 0))
    sampled_indices = []
    if args.templates_in:
        if args.fit_eeg:
            parser.error("--templates-in cannot be combined with --fit-eeg.")
        templates = np.load(args.templates_in)
        clustering_diagnostics = {"templates_in": str(args.templates_in.resolve())}
    else:
        if not args.fit_eeg:
            parser.error("Either --fit-eeg or --templates-in is required.")
        rng = np.random.default_rng(args.random_state)
        base, remainder = divmod(args.max_fit_trials, len(args.fit_eeg))
        blocks = []
        for index, eeg_path in enumerate(args.fit_eeg):
            print(f"Memory-mapping template-fit EEG: {eeg_path}")
            fit_array = np.load(eeg_path, mmap_mode="r")
            fit_trials, _ = flatten_eeg_file(fit_array)
            quota = base + (index < remainder)
            take = len(fit_trials) if args.max_fit_trials <= 0 else min(quota, len(fit_trials))
            indices = np.sort(rng.choice(len(fit_trials), take, replace=False))
            blocks.append(np.asarray(fit_trials[indices]))
            fit_sources.append({"path": str(eeg_path.resolve()), "available_trials": len(fit_trials), "sampled_trials": len(indices)})
        sampled_trials = np.concatenate(blocks)
        print(f"Using {len(sampled_trials)} balanced trials from {len(blocks)} files.")
        peak_maps, peak_sources, peak_gfp = collect_gfp_peak_maps(
            sampled_trials, sampling_rate=args.sampling_rate,
            min_peak_distance_ms=args.min_peak_distance_ms,
            max_peaks_per_trial=args.max_peaks_per_trial,
            gfp_percentile=args.gfp_percentile,
        )
        templates, clustering_diagnostics = fit_polarity_invariant_microstates(
            peak_maps, peak_gfp=peak_gfp, n_states=args.n_states,
            n_init=args.n_init, random_state=args.random_state,
        )
        np.save(args.output_dir / "microstate_templates.npy", templates)

    split_diagnostics = {}
    if not args.fit_only:
        if args.apply_eeg is None:
            parser.error("--apply-eeg is required unless --fit-only is used.")
        split_diagnostics["test"] = backfit_and_save_dataset(
            eeg_path=args.apply_eeg,
            output_dir=args.output_dir,
            output_prefix="test",
            templates=templates,
            n_states=args.n_states,
            sampling_rate=args.sampling_rate,
            min_segment_ms=args.min_segment_ms,
            batch_size=args.backfit_batch_size,
        )
    if args.backfit_train:
        split_diagnostics["train"] = backfit_and_save_dataset(
            eeg_path=args.fit_eeg[0],
            output_dir=args.output_dir,
            output_prefix="train",
            templates=templates,
            n_states=args.n_states,
            sampling_rate=args.sampling_rate,
            min_segment_ms=args.min_segment_ms,
            batch_size=args.backfit_batch_size,
        )

    metadata = {
        "fit_eeg": fit_sources,
        "apply_eeg": str(args.apply_eeg.resolve()) if args.apply_eeg else None,
        "n_states": args.n_states,
        "sampling_rate": args.sampling_rate,
        "max_fit_trials": args.max_fit_trials,
        "sampled_fit_trials": int(len(sampled_trials)),
        "max_peaks_per_trial": args.max_peaks_per_trial,
        "min_peak_distance_ms": args.min_peak_distance_ms,
        "gfp_percentile": args.gfp_percentile,
        "n_init": args.n_init,
        "min_segment_ms": args.min_segment_ms,
        "backfit_batch_size": args.backfit_batch_size,
        "random_state": args.random_state,
        "clustering_diagnostics": clustering_diagnostics,
        "split_diagnostics": split_diagnostics,
        "boundary_policy": (
            "GFP peak maps are pooled for discovery; duration, occurrence, "
            "coverage and transitions are computed separately within each trial."
        ),
    }
    with (args.output_dir / "microstate_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Templates: {templates.shape}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
