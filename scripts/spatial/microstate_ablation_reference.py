"""Reference microstate-topography ablation for a frozen NeuroBridge model.

Pipeline
--------
1. Read 63-channel, single-trial, pre-MVNN test EEG and fixed TRAIN templates.
2. Use frozen clean-data labels to define state-specific interventions:
   full topography substitution, duration shortening, occurrence removal, or
   GFP-amplitude scaling.
3. For topographic substitutions, preserve the instantaneous channel mean and
   centered L2 norm (thus GFP), with sample-count/segment-matched controls.
4. Apply the fixed TRAIN-derived MVNN matrix for the corresponding session.
5. Average the 80 test repetitions, select the channel layout required by the
   checkpoint (17 for EEGProject, 63 for TSConv), and evaluate the configured
   frozen subject model without changing its weights.

By default, every selected state-homogeneous segment is replaced by a uniformly
sampled non-winning template.  The random-time and circular-segment controls
modify exactly the same number of samples as the requested target state while
avoiding its original locations whenever possible.  Inspect the post-ablation
feature diagnostics before making neuroscientific claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ablation_PSD import (
    DEFAULT_DOWNLOAD_DIR,
    SELECTED_CHANNELS,
    checkpoint_expected_channels,
    encode_eeg,
    load_models,
    retrieval_metrics,
)
from microstate_clustering_reference import (
    backfit_microstates,
    microstate_features,
    normalize_topographies,
)


def load_session_matrices(directory: Path, sessions: int, channels: int) -> list[np.ndarray]:
    matrices = []
    for session in range(1, sessions + 1):
        path = directory / f"ses-{session:02d}_mvnn.npy"
        matrix = np.load(path)
        if matrix.shape != (channels, channels):
            raise ValueError(f"{path} has shape {matrix.shape}; expected {(channels, channels)}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path} contains non-finite values.")
        matrices.append(np.asarray(matrix, dtype=np.float64))
    return matrices


def target_runs(sequence: np.ndarray, state: int) -> list[tuple[int, int]]:
    mask = np.asarray(sequence) == state
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def build_intervention(
    trials: np.ndarray,
    labels: np.ndarray,
    templates: np.ndarray,
    state: int,
    intervention: str,
    ratio: float,
    min_segment_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build a state intervention and optional episode-level alternatives."""
    mask = np.zeros_like(labels, dtype=bool)
    alternatives = np.full_like(labels, -1, dtype=np.int16)
    eligible_episodes = 0
    selected_episodes = 0

    if intervention == "topography":
        mask = labels == state
        episode_count = sum(
            len(target_runs(sequence, state)) for sequence in labels
        )
        return mask, alternatives, {
            "eligible_episodes": episode_count,
            "selected_episodes": episode_count,
        }

    normalized_templates = normalize_topographies(templates)
    normalized_maps = normalize_topographies(np.moveaxis(trials, 1, 2))
    correlations = np.einsum(
        "ntc,kc->ntk", normalized_maps, normalized_templates, optimize=True
    )

    for trial_index, sequence in enumerate(labels):
        for start, end in target_runs(sequence, state):
            segment_length = end - start
            if intervention == "duration":
                # Shift A->B earlier, but retain at least one minimum-duration
                # A segment so that occurrence is not intentionally deleted.
                if end >= len(sequence) or segment_length <= min_segment_samples:
                    continue
                eligible_episodes += 1
                shorten = max(1, int(round(ratio * segment_length)))
                shorten = min(shorten, segment_length - min_segment_samples)
                if shorten <= 0:
                    continue
                tail_start = end - shorten
                following_state = int(sequence[end])
                mask[trial_index, tail_start:end] = True
                alternatives[trial_index, tail_start:end] = following_state
                selected_episodes += 1
            elif intervention == "occurrence":
                eligible_episodes += 1
                if rng.random() >= ratio:
                    continue
                segment_scores = np.mean(
                    np.abs(correlations[trial_index, start:end]), axis=0
                )
                segment_scores[state] = -np.inf
                alternative_state = int(np.argmax(segment_scores))
                mask[trial_index, start:end] = True
                alternatives[trial_index, start:end] = alternative_state
                selected_episodes += 1
            else:
                raise ValueError(f"Unsupported intervention: {intervention}")

    return mask, alternatives, {
        "eligible_episodes": eligible_episodes,
        "selected_episodes": selected_episodes,
    }


def relocate_mask_for_control(
    mask: np.ndarray,
    control: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Relocate a mask while matching sample count or segment structure."""
    if control == "none":
        return mask
    result = np.zeros_like(mask, dtype=bool)
    for trial_index, trial_mask in enumerate(mask):
        count = int(trial_mask.sum())
        if count == 0:
            continue
        if control == "random-time":
            candidates = np.flatnonzero(~trial_mask)
            if len(candidates) < count:
                raise ValueError(
                    "Cannot place a non-overlapping random-time control: "
                    f"need {count} samples but only {len(candidates)} are available."
                )
            selected = rng.choice(candidates, size=count, replace=False)
            result[trial_index, selected] = True
        elif control == "random-segment":
            shifts = rng.permutation(np.arange(1, len(trial_mask)))
            shifted_masks = [np.roll(trial_mask, int(shift)) for shift in shifts]
            non_overlapping = [
                shifted for shifted in shifted_masks
                if not np.any(shifted & trial_mask)
            ]
            if non_overlapping:
                result[trial_index] = non_overlapping[0]
            else:
                # Exact non-overlap is not always possible for high-coverage
                # states. Use the randomly tie-broken minimum-overlap shift.
                overlaps = np.asarray(
                    [np.count_nonzero(shifted & trial_mask) for shifted in shifted_masks]
                )
                result[trial_index] = shifted_masks[int(np.argmin(overlaps))]
        else:
            raise ValueError(f"Unsupported control: {control}")
    return result


def random_nonwinning_alternatives(
    labels: np.ndarray,
    mask: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose one random non-winning template per homogeneous mask segment."""
    if n_states < 2:
        raise ValueError("Random replacement requires at least two states.")
    alternatives = np.full_like(labels, -1, dtype=np.int16)
    for trial_index, (sequence, trial_mask) in enumerate(zip(labels, mask)):
        # Split at either a mask boundary or a clean-state transition. This
        # prevents a selected alternative from equalling the original state
        # anywhere within a relocated control segment.
        selected = np.flatnonzero(trial_mask)
        if len(selected) == 0:
            continue
        group_starts = np.r_[
            0,
            np.flatnonzero(
                (np.diff(selected) != 1)
                | (sequence[selected[1:]] != sequence[selected[:-1]])
            ) + 1,
        ]
        group_ends = np.r_[group_starts[1:], len(selected)]
        for group_start, group_end in zip(group_starts, group_ends):
            indices = selected[group_start:group_end]
            winning = int(sequence[indices[0]])
            draw = int(rng.integers(n_states - 1))
            alternative = draw + (draw >= winning)
            alternatives[trial_index, indices] = alternative
    return alternatives


def interface_taper_weights(
    mask: np.ndarray,
    smoothing_samples: int,
) -> np.ndarray:
    """Create raised-cosine weights inside every modified segment.

    The weight is zero outside the intervention mask and approaches one in
    the segment interior.  Only boundaries that touch unmodified samples are
    tapered, so no smoothing crosses an EEG-trial boundary.
    """
    mask = np.asarray(mask, dtype=bool)
    weights = mask.astype(np.float32)
    if smoothing_samples <= 0:
        return weights

    for trial_index, trial_mask in enumerate(mask):
        changes = np.diff(np.r_[False, trial_mask, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            length = int(end - start)
            if length <= 0:
                continue
            # Keep the two tapers from crossing in very short segments.
            width = min(smoothing_samples, max(1, length // 2))
            phase = np.arange(1, width + 1, dtype=np.float64) / (width + 1)
            ramp = np.sin(0.5 * np.pi * phase) ** 2
            if start > 0:
                weights[trial_index, start:start + width] = ramp
            if end < len(trial_mask):
                weights[trial_index, end - width:end] = np.minimum(
                    weights[trial_index, end - width:end], ramp[::-1]
                )
    return weights


def blend_at_interfaces(
    original: np.ndarray,
    intervened: np.ndarray,
    mask: np.ndarray,
    smoothing_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fade original and intervened EEG at mask boundaries."""
    weights = interface_taper_weights(mask, smoothing_samples)
    alpha = weights[:, None, :]
    blended = original + alpha * (intervened - original)
    return blended.astype(np.float32, copy=False), weights


def replace_with_best_alternative_topography(
    trials: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    templates: np.ndarray,
    forced_alternatives: np.ndarray | None = None,
    smoothing_samples: int = 0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace selected maps with the best non-winning template.

    The common-mode channel mean and centered spatial norm are retained at
    every time point.  Template polarity is selected from signed correlation.
    """
    output = np.asarray(trials, dtype=np.float32).copy()
    maps = np.moveaxis(output, 1, 2)  # trial, time, channel
    means = maps.mean(axis=-1, keepdims=True)
    centered = maps - means
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    unit_maps = centered / np.maximum(norms, eps)
    normalized_templates = normalize_topographies(templates)
    correlations = np.einsum("ntc,kc->ntk", unit_maps, normalized_templates)

    selected_trial, selected_time = np.nonzero(mask & (norms[..., 0] > eps))
    if len(selected_trial) == 0:
        return output, np.zeros_like(mask, dtype=np.float32)

    selected_correlations = correlations[selected_trial, selected_time].copy()
    winning_state = labels[selected_trial, selected_time]
    selected_correlations[
        np.arange(len(selected_correlations)), winning_state
    ] = 0.0
    alternative = np.argmax(np.abs(selected_correlations), axis=1)
    if forced_alternatives is not None:
        forced = forced_alternatives[selected_trial, selected_time]
        use_forced = forced >= 0
        alternative[use_forced] = forced[use_forced]
    signed_match = selected_correlations[
        np.arange(len(selected_correlations)), alternative
    ]
    polarity = np.where(signed_match < 0.0, -1.0, 1.0)

    replacements = (
        means[selected_trial, selected_time]
        + norms[selected_trial, selected_time]
        * polarity[:, None]
        * normalized_templates[alternative]
    )
    maps[selected_trial, selected_time] = replacements.astype(np.float32)
    return blend_at_interfaces(
        np.asarray(trials, dtype=np.float32), output, mask, smoothing_samples
    )


def scale_gfp_amplitude(
    trials: np.ndarray,
    mask: np.ndarray,
    scale: float,
    smoothing_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale centered scalp amplitude while retaining the channel mean."""
    output = np.asarray(trials, dtype=np.float32).copy()
    maps = np.moveaxis(output, 1, 2)
    means = maps.mean(axis=-1, keepdims=True)
    centered = maps - means
    scaled = means + scale * centered
    maps[mask] = scaled[mask]
    return blend_at_interfaces(
        np.asarray(trials, dtype=np.float32), output, mask, smoothing_samples
    )


def apply_session_mvnn(
    trials: np.ndarray,
    flat_trial_indices: np.ndarray,
    repetitions: int,
    repetitions_per_session: int,
    matrices: list[np.ndarray],
) -> np.ndarray:
    output = np.empty_like(trials, dtype=np.float32)
    repetition_indices = flat_trial_indices % repetitions
    session_indices = repetition_indices // repetitions_per_session
    if session_indices.max(initial=0) >= len(matrices):
        raise ValueError("Trial/session mapping exceeds the supplied MVNN matrices.")

    for session_index, matrix in enumerate(matrices):
        keep = session_indices == session_index
        if not np.any(keep):
            continue
        session_trials = trials[keep]
        whitened = np.einsum(
            "ntc,cd->ntd",
            np.moveaxis(session_trials, 1, 2),
            matrix,
            optimize=True,
        )
        output[keep] = np.moveaxis(whitened, 1, 2).astype(np.float32)
    return output


def load_average_selected_eeg(
    path: Path,
    selected_indices: list[int],
    batch_examples: int = 10,
) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 5:
        raise ValueError(f"Expected official test EEG with 5 dimensions, got {array.shape}.")
    output = np.empty(
        (array.shape[0] * array.shape[1], len(selected_indices), array.shape[-1]),
        dtype=np.float32,
    )
    cursor = 0
    for start in range(0, array.shape[0], batch_examples):
        block = np.asarray(array[start:start + batch_examples], dtype=np.float32)
        averaged = block.mean(axis=2)
        averaged = np.take(averaged, selected_indices, axis=-2)
        flat = averaged.reshape(-1, len(selected_indices), array.shape[-1])
        output[cursor:cursor + len(flat)] = flat
        cursor += len(flat)
    return output


def evaluate(
    eeg: np.ndarray,
    raw_image_features: np.ndarray,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    # Selecting the 17 channels can produce a non-contiguous NumPy view. Both
    # supported encoders use Tensor.view(), which requires contiguous storage.
    eeg = np.ascontiguousarray(eeg, dtype=np.float32)
    raw_image_features = np.ascontiguousarray(
        raw_image_features, dtype=np.float32
    )
    model, eeg_projector, image_projector = load_models(
        checkpoint, raw_image_features.shape[-1], device
    )
    image_tensor = torch.from_numpy(raw_image_features).to(device)
    with torch.inference_mode():
        projected_images = image_projector(image_tensor).cpu()
    eeg_features = encode_eeg(eeg, model, eeg_projector, device, batch_size)
    correct_columns = torch.arange(len(eeg), dtype=torch.long)
    return retrieval_metrics(eeg_features, projected_images, correct_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-mvnn-test",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn/sub-02/test.npy"),
    )
    parser.add_argument(
        "--pre-mvnn-info",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn/info.json"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "results/microstate_sub02_no_mvnn_k4/test_microstate_labels.npy"
        ),
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(
            "results/microstate_sub02_no_mvnn_k4/microstate_templates.npy"
        ),
    )
    parser.add_argument(
        "--mvnn-matrix-dir",
        type=Path,
        default=Path("results_sub02/mvnn_matrices_sub02"),
    )
    parser.add_argument(
        "--official-mvnn-test",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR / "things_eeg/preprocessed_eeg/sub-02/test.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR / "intra-subjects-seed2025/sub-02-seed2025/checkpoint_test_best.pth",
        help=(
            "Frozen NeuroBridge checkpoint containing model_state_dict, "
            "eeg_projector_state_dict, and img_projector_state_dict."
        ),
    )
    parser.add_argument(
        "--image-features",
        type=Path,
        default=(
            DEFAULT_DOWNLOAD_DIR
            / "things_eeg/image_feature/RN50"
            / "GaussianBlur-GaussianNoise-LowResolution-Mosaic/test.npy"
        ),
        help=(
            "Test image gallery. The default is the combined augmented feature "
            "used by scripts/things_eeg/intra-subjects.sh. Pass RN50/image_test.npy "
            "to evaluate against the unaugmented gallery."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=["clean", "state", "topography", "duration", "occurrence", "amplitude"],
        default="clean",
        help=(
            "state/topography: replace every target-state map; duration: shorten "
            "each eligible target episode; occurrence: remove a fraction of whole "
            "target episodes; amplitude: scale target-state GFP."
        ),
    )
    parser.add_argument(
        "--state",
        type=int,
        default=0,
        help="Zero-based target state. State 1 in a figure is --state 0.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help=(
            "Duration-shortening ratio or occurrence-removal probability, in "
            "(0, 1]. For example, 0.25 means 25%%."
        ),
    )
    parser.add_argument(
        "--gfp-scale",
        type=float,
        default=0.5,
        help="Centered-amplitude multiplier used by --condition amplitude.",
    )
    parser.add_argument(
        "--control",
        choices=["none", "random-time", "random-segment"],
        default="none",
        help=(
            "Relocate the intervention mask while preserving its sample count "
            "or approximate segment structure."
        ),
    )
    parser.add_argument(
        "--alternative-selection",
        choices=["random", "best"],
        default="random",
        help=(
            "How replacement templates are chosen. random samples one uniformly "
            "from the non-winning states per homogeneous segment; best uses the "
            "highest absolute spatial correlation. A target duration intervention "
            "still uses its following state by definition."
        ),
    )
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--repetitions-per-session", type=int, default=20)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--min-segment-ms", type=float, default=20.0)
    parser.add_argument(
        "--interface-smoothing-ms",
        type=float,
        default=0.0,
        help=(
            "Raised-cosine cross-fade length at each intervention boundary. "
            "Use 20 ms for the smoothed-interface protocol; 0 preserves the "
            "original hard replacement."
        ),
    )
    parser.add_argument("--processing-batch-size", type=int, default=128)
    parser.add_argument("--model-batch-size", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=2025)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/microstate_ablation_sub08"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.pre_mvnn_info.open("r", encoding="utf-8") as file:
        eeg_info = json.load(file)
    channel_names = eeg_info["ch_names"]
    expected_channels = checkpoint_expected_channels(args.checkpoint)
    if expected_channels == len(SELECTED_CHANNELS):
        selected_indices = [channel_names.index(name) for name in SELECTED_CHANNELS]
    elif expected_channels == len(channel_names):
        selected_indices = list(range(len(channel_names)))
    else:
        raise ValueError(
            f"Checkpoint expects {expected_channels} EEG channels, but the data "
            f"provides {len(channel_names)} and the configured subset has "
            f"{len(SELECTED_CHANNELS)}."
        )

    source_array = np.load(args.pre_mvnn_test, mmap_mode="r")
    labels_array = np.load(args.labels, mmap_mode="r")
    templates = np.load(args.templates)
    if source_array.shape[:-2] + (source_array.shape[-1],) != labels_array.shape:
        raise ValueError(
            f"EEG {source_array.shape} and labels {labels_array.shape} do not align."
        )
    if templates.shape != (4, source_array.shape[-2]):
        raise ValueError(
            f"This reference expects K=4 and {source_array.shape[-2]} channels; "
            f"templates are {templates.shape}."
        )
    if not 0 <= args.state < len(templates):
        raise ValueError(f"--state must be in 0..{len(templates) - 1}.")
    if args.condition in {"duration", "occurrence"} and not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio must be in (0, 1] for duration/occurrence.")
    if args.condition == "amplitude" and args.gfp_scale < 0.0:
        raise ValueError("--gfp-scale must be non-negative.")
    if args.condition == "clean" and args.control != "none":
        raise ValueError("Clean condition cannot use a matched control.")
    if args.interface_smoothing_ms < 0.0:
        raise ValueError("--interface-smoothing-ms must be non-negative.")

    leading_shape = source_array.shape[:-2]
    repetitions = leading_shape[-1]
    expected_repetitions = args.sessions * args.repetitions_per_session
    if repetitions != expected_repetitions:
        raise ValueError(
            f"Found {repetitions} repetitions; expected {expected_repetitions} "
            "from sessions * repetitions-per-session."
        )
    flat_source = source_array.reshape(-1, source_array.shape[-2], source_array.shape[-1])
    flat_labels = labels_array.reshape(-1, labels_array.shape[-1])
    matrices = load_session_matrices(
        args.mvnn_matrix_dir, args.sessions, source_array.shape[-2]
    )

    n_examples = int(np.prod(leading_shape[:-1]))
    whitened_sum = np.zeros(
        (n_examples, source_array.shape[-2], source_array.shape[-1]), dtype=np.float64
    )
    counts = np.zeros(n_examples, dtype=np.int64)
    rng = np.random.default_rng(args.random_state)
    modified_samples = 0
    total_samples = 0
    eligible_episodes = 0
    selected_episodes = 0

    feature_sums: dict[str, np.ndarray] = {}
    feature_trials = 0
    gev_numerator = 0.0
    gev_denominator = 0.0
    correlation_sum = 0.0
    correlation_count = 0
    original_energy = 0.0
    altered_energy = 0.0

    min_segment_samples = max(
        1, int(round(args.min_segment_ms * args.sampling_rate / 1000.0))
    )
    smoothing_samples = max(
        0, int(round(args.interface_smoothing_ms * args.sampling_rate / 1000.0))
    )
    taper_weight_sum = 0.0
    for start in range(0, len(flat_source), args.processing_batch_size):
        end = min(start + args.processing_batch_size, len(flat_source))
        trials = np.asarray(flat_source[start:end], dtype=np.float32)
        labels = np.asarray(flat_labels[start:end], dtype=np.int64)

        intervention = "topography" if args.condition == "state" else args.condition
        if intervention == "clean":
            altered = trials.copy()
            mask = np.zeros_like(labels, dtype=bool)
            taper_weights = np.zeros_like(labels, dtype=np.float32)
            episode_counts = {"eligible_episodes": 0, "selected_episodes": 0}
        elif intervention == "amplitude":
            mask = labels == args.state
            episode_counts = {
                "eligible_episodes": sum(
                    len(target_runs(sequence, args.state)) for sequence in labels
                ),
                "selected_episodes": sum(
                    len(target_runs(sequence, args.state)) for sequence in labels
                ),
            }
            if args.control != "none":
                mask = relocate_mask_for_control(mask, args.control, rng)
            altered, taper_weights = scale_gfp_amplitude(
                trials,
                mask,
                args.gfp_scale,
                smoothing_samples=smoothing_samples,
            )
        else:
            mask, alternatives, episode_counts = build_intervention(
                trials=trials,
                labels=labels,
                templates=templates,
                state=args.state,
                intervention=intervention,
                ratio=args.ratio,
                min_segment_samples=min_segment_samples,
                rng=rng,
            )
            if args.control != "none":
                mask = relocate_mask_for_control(mask, args.control, rng)
                # Recompute alternatives for the states actually winning at
                # each newly selected control location.
                alternatives = np.full_like(labels, -1, dtype=np.int16)
            if args.alternative_selection == "random" and (
                intervention in {"topography", "occurrence"}
                or args.control != "none"
            ):
                alternatives = random_nonwinning_alternatives(
                    labels, mask, len(templates), rng
                )
            altered, taper_weights = replace_with_best_alternative_topography(
                trials,
                labels,
                mask,
                templates,
                forced_alternatives=alternatives,
                smoothing_samples=smoothing_samples,
            )
        eligible_episodes += episode_counts["eligible_episodes"]
        selected_episodes += episode_counts["selected_episodes"]

        new_labels, winning = backfit_microstates(
            altered, templates, min_segment_samples=min_segment_samples
        )
        features = microstate_features(
            new_labels,
            n_states=len(templates),
            sampling_rate=args.sampling_rate,
            trials=altered,
        )
        for name, values in features.items():
            values = np.asarray(values, dtype=np.float64)
            batch_sum = values.sum(axis=0)
            feature_sums[name] = feature_sums.get(name, np.zeros_like(batch_sum)) + batch_sum
        feature_trials += len(altered)

        gfp_squared = np.std(altered, axis=1, dtype=np.float64) ** 2
        gev_numerator += float(np.sum(gfp_squared * winning.astype(np.float64) ** 2))
        gev_denominator += float(gfp_squared.sum())
        correlation_sum += float(np.abs(winning).sum())
        correlation_count += winning.size
        original_energy += float(np.square(trials.astype(np.float64)).sum())
        altered_energy += float(np.square(altered.astype(np.float64)).sum())
        modified_samples += int(mask.sum())
        taper_weight_sum += float(taper_weights.sum())
        total_samples += mask.size

        flat_indices = np.arange(start, end, dtype=np.int64)
        whitened = apply_session_mvnn(
            altered,
            flat_indices,
            repetitions,
            args.repetitions_per_session,
            matrices,
        )
        example_indices = flat_indices // repetitions
        np.add.at(whitened_sum, example_indices, whitened.astype(np.float64))
        np.add.at(counts, example_indices, 1)

    if np.any(counts != repetitions):
        raise RuntimeError("Not every test example received all repetitions.")
    reconstructed = (whitened_sum / counts[:, None, None]).astype(np.float32)
    reconstructed_selected = reconstructed[:, selected_indices]

    official_selected = load_average_selected_eeg(
        args.official_mvnn_test, selected_indices
    )
    raw_image_features = np.load(args.image_features).reshape(
        -1, np.load(args.image_features, mmap_mode="r").shape[-1]
    ).astype(np.float32)
    if len(raw_image_features) != len(reconstructed_selected):
        raise ValueError("Image gallery and reconstructed EEG have different lengths.")

    device = torch.device(args.device)
    official_metrics = evaluate(
        official_selected,
        raw_image_features,
        args.checkpoint,
        device,
        args.model_batch_size,
    )
    condition_metrics = evaluate(
        reconstructed_selected,
        raw_image_features,
        args.checkpoint,
        device,
        args.model_batch_size,
    )

    difference = reconstructed_selected.astype(np.float64) - official_selected
    clean_comparison = {
        "rmse_against_official_mvnn_average": float(np.sqrt(np.mean(difference ** 2))),
        "relative_rmse": float(
            np.sqrt(np.mean(difference ** 2))
            / np.maximum(np.sqrt(np.mean(official_selected.astype(np.float64) ** 2)), 1e-12)
        ),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }
    diagnostics = {
        "modified_sample_fraction": modified_samples / max(total_samples, 1),
        "effective_weighted_modified_fraction": taper_weight_sum
        / max(total_samples, 1),
        "interface_smoothing_ms": args.interface_smoothing_ms,
        "interface_smoothing_samples": smoothing_samples,
        "eligible_target_episodes": eligible_episodes,
        "selected_target_episodes": selected_episodes,
        "selected_episode_fraction": selected_episodes
        / max(eligible_episodes, 1),
        "mean_absolute_winning_spatial_correlation": correlation_sum
        / max(correlation_count, 1),
        "global_explained_variance": gev_numerator / max(gev_denominator, 1e-12),
        "rms_ratio_pre_mvnn": np.sqrt(altered_energy / max(original_energy, 1e-12)),
        "mean_features": {
            name: (values / feature_trials).tolist()
            for name, values in feature_sums.items()
        },
    }
    payload = {
        "condition": args.condition,
        "intervention": (
            "topography" if args.condition == "state" else args.condition
        ),
        "control": args.control,
        "alternative_selection": args.alternative_selection,
        "interface_smoothing_ms": args.interface_smoothing_ms,
        "ratio": args.ratio if args.condition in {"duration", "occurrence"} else None,
        "gfp_scale": args.gfp_scale if args.condition == "amplitude" else None,
        "target_state_zero_based": args.state,
        "target_state_display": args.state + 1,
        "random_state": args.random_state,
        "templates": str(args.templates.resolve()),
        "labels": str(args.labels.resolve()),
        "mvnn_matrix_dir": str(args.mvnn_matrix_dir.resolve()),
        "image_features": str(args.image_features.resolve()),
        "official_metrics": official_metrics,
        "condition_metrics": condition_metrics,
        "drop_relative_to_official": {
            "top1": official_metrics["top1"] - condition_metrics["top1"],
            "top5": official_metrics["top5"] - condition_metrics["top5"],
        },
        "clean_reconstruction_comparison": clean_comparison,
        "post_condition_diagnostics": diagnostics,
        "important_note": (
            "For an ablated condition, the numerical comparison against official "
            "MVNN is not a clean-reconstruction error. Run --condition clean first."
        ),
    }

    if args.condition == "clean":
        output_stem = "clean"
    else:
        intervention_name = (
            "topography" if args.condition == "state" else args.condition
        )
        dose = (
            f"ratio-{args.ratio:g}"
            if intervention_name in {"duration", "occurrence"}
            else f"scale-{args.gfp_scale:g}"
            if intervention_name == "amplitude"
            else "full"
        )
        output_stem = (
            f"{intervention_name}_state-{args.state + 1}_{dose}"
            f"_control-{args.control}_alternative-{args.alternative_selection}"
            f"_smooth-{args.interface_smoothing_ms:g}ms"
            f"_seed-{args.random_state}"
        )
    json_path = args.output_dir / f"{output_stem}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    # Keep the randomized-alternative protocol separate from CSV files made by
    # the earlier best-alternative implementation, whose schema and estimand
    # differ.
    csv_path = args.output_dir / "microstate_random_alternative_ablation_results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "condition", "control", "alternative_selection",
            "interface_smoothing_ms", "state",
            "ratio", "gfp_scale", "seed",
            "modified_fraction", "eligible_episodes", "selected_episodes",
            "selected_episode_fraction", "effective_weighted_modified_fraction",
            "top1", "top5", "top1_drop_vs_official", "top5_drop_vs_official",
            "mean_abs_r", "gev", "rms_ratio_pre_mvnn",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "condition": args.condition,
                "control": args.control,
                "alternative_selection": args.alternative_selection,
                "interface_smoothing_ms": args.interface_smoothing_ms,
                "state": "" if args.condition == "clean" else args.state + 1,
                "ratio": (
                    args.ratio
                    if args.condition in {"duration", "occurrence"}
                    else ""
                ),
                "gfp_scale": args.gfp_scale if args.condition == "amplitude" else "",
                "seed": args.random_state,
                "modified_fraction": diagnostics["modified_sample_fraction"],
                "eligible_episodes": diagnostics["eligible_target_episodes"],
                "selected_episodes": diagnostics["selected_target_episodes"],
                "selected_episode_fraction": diagnostics[
                    "selected_episode_fraction"
                ],
                "effective_weighted_modified_fraction": diagnostics[
                    "effective_weighted_modified_fraction"
                ],
                "top1": condition_metrics["top1"],
                "top5": condition_metrics["top5"],
                "top1_drop_vs_official": payload["drop_relative_to_official"]["top1"],
                "top5_drop_vs_official": payload["drop_relative_to_official"]["top5"],
                "mean_abs_r": diagnostics[
                    "mean_absolute_winning_spatial_correlation"
                ],
                "gev": diagnostics["global_explained_variance"],
                "rms_ratio_pre_mvnn": diagnostics["rms_ratio_pre_mvnn"],
            }
        )

    print(
        f"Official: Top-1={official_metrics['top1']:.4f}, "
        f"Top-5={official_metrics['top5']:.4f}"
    )
    print(
        f"{output_stem}: Top-1={condition_metrics['top1']:.4f}, "
        f"Top-5={condition_metrics['top5']:.4f}, "
        f"modified={diagnostics['modified_sample_fraction']:.4f}"
    )
    if args.condition == "clean":
        print(
            "Clean reconstruction relative RMSE: "
            f"{clean_comparison['relative_rmse']:.6f}"
        )
    print(f"Saved {json_path}")
    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()
