"""Download intra/inter seed9999 checkpoints from junmoxiao-cloud/2026BMI-."""
from __future__ import annotations

import os
import shutil
import time
import urllib.request
from pathlib import Path

REPO = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
ROOT = Path(r"D:\NeuroBridge\checkpoints\seed9999")

INTRA_RUNS = {
    1: "NeuroBridge/9999-intra-subjects/20260728-191248-sub-01/checkpoint_test_best.pth",
    2: "NeuroBridge/9999-intra-subjects/20260728-191556-sub-02/checkpoint_test_best.pth",
    3: "NeuroBridge/9999-intra-subjects/20260728-191913-sub-03/checkpoint_test_best.pth",
    4: "NeuroBridge/9999-intra-subjects/20260728-192218-sub-04/checkpoint_test_best.pth",
    5: "NeuroBridge/9999-intra-subjects/20260728-192522-sub-05/checkpoint_test_best.pth",
    6: "NeuroBridge/9999-intra-subjects/20260728-192834-sub-06/checkpoint_test_best.pth",
    7: "NeuroBridge/9999-intra-subjects/20260728-193145-sub-07/checkpoint_test_best.pth",
    8: "NeuroBridge/9999-intra-subjects/20260728-193502-sub-08/checkpoint_test_best.pth",
    9: "NeuroBridge/9999-intra-subjects/20260728-193823-sub-09/checkpoint_test_best.pth",
    10: "NeuroBridge/9999-intra-subjects/20260728-194145-sub-10/checkpoint_test_best.pth",
}

INTER_RUNS = {
    1: "NeuroBridge/9999-inter-subjects/20260728-200756-sub-01/checkpoint_test_best.pth",
    2: "NeuroBridge/9999-inter-subjects/20260728-214203-sub-02/checkpoint_test_best.pth",
    3: "NeuroBridge/9999-inter-subjects/20260728-231748-sub-03/checkpoint_test_best.pth",
    4: "NeuroBridge/9999-inter-subjects/20260729-005036-sub-04/checkpoint_test_best.pth",
    5: "NeuroBridge/9999-inter-subjects/20260729-022034-sub-05/checkpoint_test_best.pth",
    6: "NeuroBridge/9999-inter-subjects/20260729-035153-sub-06/checkpoint_test_best.pth",
    7: "NeuroBridge/9999-inter-subjects/20260729-052319-sub-07/checkpoint_test_best.pth",
    8: "NeuroBridge/9999-inter-subjects/20260729-065345-sub-08/checkpoint_test_best.pth",
    9: "NeuroBridge/9999-inter-subjects/20260729-082505-sub-09/checkpoint_test_best.pth",
    10: "NeuroBridge/9999-inter-subjects/20260729-095434-sub-10/checkpoint_test_best.pth",
}


def is_real(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    with path.open("rb") as f:
        return not f.read(80).startswith(b"version https://git-lfs.github.com")


def download(rel: str, dst: Path, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_real(dst):
        print(f"SKIP {dst} ({dst.stat().st_size/1e6:.1f}MB)", flush=True)
        return
    url = f"{REPO}/{rel.replace(os.sep, '/')}"
    print(f"GET {url}", flush=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            resume = tmp.stat().st_size if tmp.is_file() else 0
            headers = {"User-Agent": "NeuroBridge/seed9999"}
            if resume:
                headers["Range"] = f"bytes={resume}-"
                print(f"  resume from {resume/1e6:.1f}MB", flush=True)
            req = urllib.request.Request(url, headers=headers)
            chunk = 1024 * 1024
            with urllib.request.urlopen(req, timeout=1800) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                mode = "ab" if resume else "wb"
                if resume and code == 200:
                    print("  server ignored Range, restart from 0", flush=True)
                    mode = "wb"
                    resume = 0
                n = resume
                milestone = n // (10 * chunk)
                with tmp.open(mode) as out:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        out.write(buf)
                        n += len(buf)
                        m = n // (10 * chunk)
                        if m > milestone:
                            print(f"  ... {n/1e6:.1f}MB", flush=True)
                            milestone = m
            size = tmp.stat().st_size
            if size < 1_000_000:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"too small ({size} bytes), likely LFS pointer")
            tmp.replace(dst)
            print(f"  OK {dst} ({size/1e6:.1f}MB)", flush=True)
            return
        except Exception as e:
            last_err = e
            print(f"  FAIL attempt {attempt}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
    assert last_err is not None
    raise last_err


def main() -> int:
    ok = fail = 0
    for sid, rel in INTRA_RUNS.items():
        dst = ROOT / "intra" / f"sub-{sid:02d}-seed9999" / "checkpoint_test_best.pth"
        try:
            download(rel, dst)
            ok += 1
        except Exception as e:
            print(f"FAIL intra sub-{sid:02d}: {e}", flush=True)
            fail += 1

    for sid, rel in INTER_RUNS.items():
        dst = ROOT / "inter" / f"sub-{sid:02d}-seed9999" / "checkpoint_test_best.pth"
        try:
            download(rel, dst)
            ok += 1
        except Exception as e:
            print(f"FAIL inter sub-{sid:02d}: {e}", flush=True)
            fail += 1

    print(f"\nDONE ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
