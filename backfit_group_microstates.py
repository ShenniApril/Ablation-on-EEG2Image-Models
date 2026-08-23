"""Back-fit frozen group microstate templates to multiple subjects' test EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microstate_clustering_reference import backfit_and_save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eeg-root",
        type=Path,
        default=Path("data/things_eeg/preprocessed_eeg_no_mvnn"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--min-segment-ms", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if len(set(args.subjects)) != len(args.subjects):
        raise ValueError("--subjects contains duplicates.")
    templates = np.load(args.templates)
    if templates.ndim != 2:
        raise ValueError(f"Expected templates shaped (states, channels), got {templates.shape}")
    n_states = int(templates.shape[0])
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_diagnostics: dict[str, object] = {}

    for subject in args.subjects:
        name = f"sub-{subject:02d}"
        eeg_path = args.eeg_root / name / "test.npy"
        if not eeg_path.is_file():
            raise FileNotFoundError(f"Missing test EEG for {name}: {eeg_path}")
        output_dir = args.output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = backfit_and_save_dataset(
            eeg_path=eeg_path,
            output_dir=output_dir,
            output_prefix="test",
            templates=templates,
            n_states=n_states,
            sampling_rate=args.sampling_rate,
            min_segment_ms=args.min_segment_ms,
            batch_size=args.batch_size,
        )
        metadata = {
            "subject": name,
            "templates": str(args.templates.resolve()),
            "template_shape": list(templates.shape),
            "sampling_rate": args.sampling_rate,
            "min_segment_ms": args.min_segment_ms,
            "split_diagnostics": {"test": diagnostics},
        }
        with (output_dir / "microstate_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        all_diagnostics[name] = diagnostics
        print(f"Saved {name} back-fit outputs to: {output_dir}")

    with (args.output_root / "group_backfit_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "templates": str(args.templates.resolve()),
                "subjects": [f"sub-{subject:02d}" for subject in args.subjects],
                "subject_diagnostics": all_diagnostics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
