"""Generate zero/noise EEG (test split) and eval ori/zero/noise for seed1234 intra+inter."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
SEED = 1234
CKPT_INTRA = os.path.join(_REPO, "data", "checkpoints", "seed1234", "intra")
CKPT_INTER = os.path.join(_REPO, "data", "checkpoints", "seed1234", "inter")


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


def prep_subject(sid: int) -> None:
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
    parser.add_argument(
        "--mode",
        choices=["intra", "inter", "both"],
        default="both",
    )
    parser.add_argument(
        "--sub_ids",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
    )
    parser.add_argument(
        "--skip_prep",
        action="store_true",
        help="skip zero/amp/gauss generation (already on disk)",
    )
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
