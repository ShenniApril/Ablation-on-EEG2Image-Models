"""Download intra/inter seed3407 checkpoints from junmoxiao-cloud/2026BMI-/results."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

REPO = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
ROOT = Path(r"D:\NeuroBridge\checkpoints\seed3407")

EXPECTED_INTRA = 77_464_705
EXPECTED_INTER = 44_193_613
TOL = 0.15


def rel_intra(sid: int) -> str:
    return f"results/intra-subjects-seed3407/sub-{sid:02d}-seed3407/checkpoint_test_best.pth"


def rel_inter(sid: int) -> str:
    return f"results/inter-subjects-seed3407/sub-{sid:02d}-seed3407/checkpoint_test_best.pth"


def is_real(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    lo = int(expected * (1 - TOL))
    hi = int(expected * (1 + TOL))
    if size < lo or size > hi:
        return False
    with path.open("rb") as f:
        return not f.read(80).startswith(b"version https://git-lfs.github.com")


def download(rel: str, dst: Path, expected: int, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_real(dst, expected):
        print(f"SKIP {dst} ({dst.stat().st_size/1e6:.1f}MB)", flush=True)
        return
    if dst.is_file() and not is_real(dst, expected):
        print(f"REMOVE bad {dst} ({dst.stat().st_size} bytes)", flush=True)
        dst.unlink(missing_ok=True)
    url = f"{REPO}/{rel.replace(os.sep, '/')}"
    print(f"GET {url}", flush=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            resume = tmp.stat().st_size if tmp.is_file() else 0
            headers = {"User-Agent": "NeuroBridge/seed3407"}
            if resume:
                headers["Range"] = f"bytes={resume}-"
            req = urllib.request.Request(url, headers=headers)
            chunk = 1024 * 1024
            with urllib.request.urlopen(req, timeout=1800) as resp:
                mode = "ab" if resume else "wb"
                if resume and (getattr(resp, "status", None) or resp.getcode()) == 200:
                    mode = "wb"
                    resume = 0
                with tmp.open(mode) as out:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        out.write(buf)
            size = tmp.stat().st_size
            if not is_real(tmp, expected):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"bad size {size}, expected ~{expected}")
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
    for sid in range(1, 11):
        dst = ROOT / "intra" / f"sub-{sid:02d}-seed3407" / "checkpoint_test_best.pth"
        try:
            download(rel_intra(sid), dst, EXPECTED_INTRA)
            ok += 1
        except Exception as e:
            print(f"FAIL intra sub-{sid:02d}: {e}", flush=True)
            fail += 1
    for sid in range(1, 11):
        dst = ROOT / "inter" / f"sub-{sid:02d}-seed3407" / "checkpoint_test_best.pth"
        try:
            download(rel_inter(sid), dst, EXPECTED_INTER)
            ok += 1
        except Exception as e:
            print(f"FAIL inter sub-{sid:02d}: {e}", flush=True)
            fail += 1
    print(f"\nDONE ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
