"""Download intra/inter seed1234 checkpoints from junmoxiao-cloud/2026BMI-."""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
ROOT = Path(r"D:\git\NeuroBridge\data\checkpoints\seed1234")

INTRA = [
    (sid, f"results/intra-subjects-seed1234/sub-{sid:02d}-seed1234/checkpoint_test_best.pth")
    for sid in range(1, 11)
]

INTER_RUNS = {
    1: "NeuroBridge/1234-inter-subjects/20260730-002658-sub-01/checkpoint_test_best.pth",
    2: "NeuroBridge/1234-inter-subjects/20260730-015822-sub-02/checkpoint_test_best.pth",
    3: "NeuroBridge/1234-inter-subjects/20260730-033041-sub-03/checkpoint_test_best.pth",
    4: "NeuroBridge/1234-inter-subjects/20260730-050017-sub-04/checkpoint_test_best.pth",
    5: "NeuroBridge/1234-inter-subjects/20260730-063032-sub-05/checkpoint_test_best.pth",
    6: "NeuroBridge/1234-inter-subjects/20260730-075901-sub-06/checkpoint_test_best.pth",
    7: "NeuroBridge/1234-inter-subjects/20260730-093016-sub-07/checkpoint_test_best.pth",
    8: "NeuroBridge/1234-inter-subjects/20260730-110138-sub-08/checkpoint_test_best.pth",
    9: "NeuroBridge/1234-inter-subjects/20260730-123533-sub-09/checkpoint_test_best.pth",
    10: "NeuroBridge/1234-inter-subjects/20260730-140806-sub-10/checkpoint_test_best.pth",
}


def is_real(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    with path.open("rb") as f:
        return not f.read(80).startswith(b"version https://git-lfs.github.com")


def download(rel: str, dst: Path, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_real(dst):
        print(f"SKIP {dst.name} ({dst.stat().st_size/1e6:.1f}MB)", flush=True)
        return
    url = f"{REPO}/{rel.replace(os.sep, '/')}"
    print(f"GET {url}", flush=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NeuroBridge/seed1234"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
            if len(data) < 1_000_000:
                raise RuntimeError(f"too small ({len(data)} bytes), likely LFS pointer")
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dst)
            print(f"  OK {dst} ({len(data)/1e6:.1f}MB)", flush=True)
            return
        except Exception as e:
            last_err = e
            print(f"  FAIL attempt {attempt}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
    assert last_err is not None
    raise last_err


def main() -> int:
    ok = fail = 0
    for sid, rel in INTRA:
        dst = ROOT / "intra" / f"sub-{sid:02d}-seed1234" / "checkpoint_test_best.pth"
        try:
            download(rel, dst)
            ok += 1
        except Exception as e:
            print(f"FAIL intra sub-{sid:02d}: {e}", flush=True)
            fail += 1

    for sid, rel in INTER_RUNS.items():
        dst = ROOT / "inter" / f"sub-{sid:02d}-seed1234" / "checkpoint_test_best.pth"
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
