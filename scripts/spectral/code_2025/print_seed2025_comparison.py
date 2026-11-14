import csv
import os
import statistics as st
from collections import defaultdict

REPO = os.path.join(os.path.dirname(__file__), "..", "data")


def load(mode: str) -> dict:
    path = os.path.join(
        REPO, f"acc_ori_zero_noise_results_{mode}_seed2025", "all_subs_ori_zero_noise_summary.csv"
    )
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by: dict = defaultdict(dict)
    for r in rows:
        by[r["sub_id"]][r["condition"]] = r
    return by


def main() -> None:
    intra = load("intra")
    inter = load("inter")
    print("=== seed2025 Original Top-1 / Top-5 (%) ===")
    print(" sub   intra_T1  intra_T5   inter_T1  inter_T5   gap_T1")
    intra_t, inter_t = [], []
    for sid in range(1, 11):
        s = f"{sid:02d}"
        oi, oe = intra[s]["original"], inter[s]["original"]
        a, b = float(oi["top1"]), float(oe["top1"])
        intra_t.append(a)
        inter_t.append(b)
        print(
            f" {s}   {a:8.1f}  {float(oi['top5']):8.1f}   {b:8.1f}  {float(oe['top5']):8.1f}   {a - b:+7.1f}"
        )
    print(
        f" mean   {st.mean(intra_t):8.1f}            {st.mean(inter_t):8.1f}            {st.mean(intra_t) - st.mean(inter_t):+7.1f}"
    )

    bands = ["delta", "theta", "alpha", "beta", "low_gamma", "gamma", "high_gamma"]
    print("\n=== Group-mean Top-1 drop when band ZEROED ===")
    for band in bands:
        di = [float(intra[f"{sid:02d}"][f"zero_{band}"]["delta_top1"]) for sid in range(1, 11)]
        de = [float(inter[f"{sid:02d}"][f"zero_{band}"]["delta_top1"]) for sid in range(1, 11)]
        print(f" {band:>12}  intra {st.mean(di):+6.1f}   inter {st.mean(de):+6.1f}")


if __name__ == "__main__":
    main()
