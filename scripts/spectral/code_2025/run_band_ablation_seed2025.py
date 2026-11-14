"""Generate noise EEG (test) and eval ori/zero/noise for seed2025 intra+inter.

Zero-band EEG is deterministic (same as other seeds). Reuses existing
sub-XX-zero-bands and amplitude&phase(orignal)_out when present.
Gaussian noise uses seed=2025 into data/gaussian-noise_out_seed2025/.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
SEED = 2025
CKPT_INTRA = r"D:\NeuroBridge\checkpoints\seed2025\intra"
CKPT_INTER = r"D:\NeuroBridge\checkpoints\seed2025\inter"
NOISE_ROOT = os.path.join(_REPO, "data", "gaussian-noise_out_seed2025")


def run(cmd: list[str], env: dict | None = None) -> int:
    print("\n>>", " ".join(cmd), flush=True)
    e = {**os.environ, "PYTHONPATH": _REPO}
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=_REPO, env=e).returncode


def ckpt_intra(sid: int) -> str:
    return os.path.join(CKPT_INTRA, f"sub-{sid:02d}-seed{SEED}", "checkpoint_test_best.pth")


def ckpt_inter(sid: int) -> str:
    return os.path.join(CKPT_INTER, f"sub-{sid:02d}-seed{SEED}", "checkpoint_test_best.pth")


def zero_bands_ready(sid: int) -> bool:
    root = os.path.join(_REPO, "data", "things_eeg", f"sub-{sid:02d}-zero-bands")
    if not os.path.isdir(root):
        return False
    for band in ("delta", "theta", "alpha", "beta", "low-gamma", "gamma", "high-gamma"):
        p = os.path.join(root, f"sub-{sid:02d}-{band}", "test.npy")
        if not os.path.isfile(p):
            return False
    return True


def amp_phase_ready(sid: int) -> bool:
    split = os.path.join(
        _REPO, "data", "amplitude&phase(orignal)_out", f"sub-{sid:02d}", "test"
    )
    return os.path.isfile(os.path.join(split, "delta_amplitude.npy"))


def noise_ready(sid: int) -> bool:
    for band in ("delta", "theta", "alpha", "beta", "low_gamma", "gamma", "high_gamma"):
        p = os.path.join(NOISE_ROOT, f"sub-{sid:02d}", band, "test.npy")
        if not os.path.isfile(p):
            return False
    return True


def prep_subject(sid: int) -> None:
    if not zero_bands_ready(sid):
        run(
            [
                PY,
                "-u",
                os.path.join("scripts", "zero_bands_subjects"),
                "--sub_ids",
                str(sid),
                "--splits",
                "test",
                "--seed",
                str(SEED),
                "--checkpoint_path",
                ckpt_intra(sid),
            ]
        )
    else:
        print(f"SKIP zero-bands sub-{sid:02d} (already on disk)", flush=True)

    if not amp_phase_ready(sid):
        run(
            [
                PY,
                "-u",
                os.path.join("data", "amplitude&phase(orignal)"),
                "--sub_id",
                str(sid),
                "--splits",
                "test",
                "--chunk_size",
                "200",
            ]
        )
    else:
        print(f"SKIP amp/phase sub-{sid:02d}", flush=True)

    if noise_ready(sid):
        print(f"SKIP gaussian-noise sub-{sid:02d} (already on disk)", flush=True)
        return

    run(
        [
            PY,
            "-u",
            os.path.join("data", "gaussian-noise"),
            "--sub_id",
            str(sid),
            "--seed",
            str(SEED),
            "--splits",
            "test",
            "--out_dir",
            NOISE_ROOT,
            "--checkpoint_path",
            ckpt_intra(sid),
        ]
    )


def eval_one(sid: int, mode: str, out_dir: str, staging: str) -> int:
    ckpt = ckpt_intra(sid) if mode == "intra" else ckpt_inter(sid)
    cmd = [
        PY,
        "-u",
        os.path.join("data", "acc_ori_zero_noise"),
        "--sub_id",
        str(sid),
        "--seed",
        str(SEED),
        "--checkpoint_path",
        ckpt,
        "--noise_root",
        NOISE_ROOT,
        "--device",
        "cpu",
        "--out_dir",
        out_dir,
        "--staging_dir",
        staging,
    ]
    if mode == "inter":
        cmd += ["--eeg_encoder_type", "TSConv", "--all_channels"]
    return run(cmd)


def summarize(out_dir: str) -> None:
    rows: list[dict] = []
    for sid in range(1, 11):
        csv_path = os.path.join(out_dir, f"sub-{sid:02d}_ori_zero_noise_acc.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["sub_id"] = f"{sid:02d}"
                rows.append(row)
    if not rows:
        print(f"No CSV rows in {out_dir}", flush=True)
        return
    summary = os.path.join(out_dir, "all_subs_ori_zero_noise_summary.csv")
    fields = ["sub_id"] + [k for k in rows[0] if k != "sub_id"]
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {summary}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["intra", "inter", "both"], default="both")
    parser.add_argument("--sub_ids", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--skip_prep", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    modes = ["intra", "inter"] if args.mode == "both" else [args.mode]

    for sid in args.sub_ids:
        if not args.skip_prep:
            print(f"\n######## PREP sub-{sid:02d} ########", flush=True)
            prep_subject(sid)

    if args.skip_eval:
        return 0

    for mode in modes:
        out_dir = os.path.join(
            _REPO, "data", f"acc_ori_zero_noise_results_{mode}_seed{SEED}"
        )
        staging = os.path.join(
            _REPO, "data", f"acc_ori_zero_noise_staging_{mode}_seed{SEED}"
        )
        os.makedirs(out_dir, exist_ok=True)
        for sid in args.sub_ids:
            if os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
            rc = eval_one(sid, mode, out_dir, staging)
            print(f"eval {mode} sub-{sid:02d} exit={rc}", flush=True)
        summarize(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
