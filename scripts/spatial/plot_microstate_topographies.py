"""Plot microstate scalp topographies with one shared symmetric color scale.

The templates produced by ``microstate_clustering_reference.py`` are
average-referenced and L2-normalized.  Consequently, the plotted values are
unitless normalized voltage, not microvolts.  Microstate polarity is
equivalent, so interpret spatial pattern rather than red/blue sign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


def load_channel_names(info_path: Path) -> list[str]:
    with info_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    channel_names = metadata.get("ch_names")
    if not isinstance(channel_names, list) or not channel_names:
        raise ValueError(f"{info_path} does not contain a valid ch_names list.")
    return channel_names


def plot_templates(
    templates: np.ndarray,
    channel_names: list[str],
    output_path: Path,
    montage_name: str = "standard_1005",
    state_prefix: str = "State",
    dpi: int = 300,
    show_channel_names: bool = False,
    sphere_radius: float = 0.12,
) -> None:
    templates = np.asarray(templates, dtype=np.float64)
    if templates.ndim != 2:
        raise ValueError(
            f"Expected templates shaped (states, channels), got {templates.shape}."
        )
    if templates.shape[1] != len(channel_names):
        raise ValueError(
            f"Templates have {templates.shape[1]} channels, but info.json lists "
            f"{len(channel_names)}."
        )
    if not np.isfinite(templates).all():
        raise ValueError("Templates contain non-finite values.")

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=250.0,
        ch_types=["eeg"] * len(channel_names),
    )
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, match_case=False, on_missing="raise")

    n_states = templates.shape[0]
    figure_width = max(9.0, 3.0 * n_states)
    fig, axes = plt.subplots(
        1,
        n_states,
        figsize=(figure_width, 3.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    # One shared scale makes the states visually comparable. Templates are
    # normalized, so this is normalized voltage rather than microvolts.
    vmax = float(np.max(np.abs(templates)))
    if vmax <= 0:
        raise ValueError("All template values are zero.")

    image = None
    for state_index, axis in enumerate(axes):
        image, _ = mne.viz.plot_topomap(
            templates[state_index],
            info,
            axes=axis,
            show=False,
            sensors=True,
            names=channel_names if show_channel_names else None,
            contours=6,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            extrapolate="head",
            sphere=(0.0, 0.0, 0.0, sphere_radius),
        )
        axis.set_title(f"{state_prefix} {state_index + 1}", fontsize=13)

    if image is None:
        raise RuntimeError("No topography was plotted.")
    colorbar = fig.colorbar(
        image,
        ax=axes.tolist(),
        orientation="horizontal",
        shrink=0.55,
        pad=0.08,
    )
    colorbar.set_label("Normalized scalp voltage (unitless)")
    fig.suptitle(
        "Subject-specific RSVP microstate templates\n"
        "(polarity-invariant; red/blue sign is arbitrary)",
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(
            "results_inter/group_microstate_k4/microstate_templates.npy"
        ),
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn/info.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/microstate_sub08_no_mvnn_k4/microstate_topographies.png"
        ),
    )
    parser.add_argument("--montage", default="standard_1005")
    parser.add_argument("--state-prefix", default="State")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-channel-names", action="store_true")
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=0.12,
        help="Head sphere radius in metres for the standard_1005 coordinates.",
    )
    args = parser.parse_args()

    templates = np.load(args.templates)
    channel_names = load_channel_names(args.info)
    plot_templates(
        templates=templates,
        channel_names=channel_names,
        output_path=args.output,
        montage_name=args.montage,
        state_prefix=args.state_prefix,
        dpi=args.dpi,
        show_channel_names=args.show_channel_names,
        sphere_radius=args.sphere_radius,
    )
    print(f"Templates: {templates.shape}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
