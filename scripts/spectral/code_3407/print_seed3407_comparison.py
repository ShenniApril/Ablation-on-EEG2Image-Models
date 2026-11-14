import csv
import os
import statistics as st
from collections import defaultdict

REPO = os.path.join(os.path.dirname(__file__), "..", "data")


def load(mode: str) -> dict:
    path = os.path.join(
        REPO, f"acc_ori_zero_noise_results_{mode}_seed3407", "all_subs_ori_zero_noise_summary.csv"
    )
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by: dict = defaultdict(dict)
    for r in rows:
        by[r["sub_id"]][r["condition"]] = r
    return by


def main() -> None:
    intra, inter = load("intra"), load("inter")
    print("=== seed3407 Original Top-1 / Top-5 (%) ===")
    print(" sub   intra_T1  intra_T5   inter_T1  inter_T5   gap_T1")
    t1i, t1e = [], []
    for sid in range(1, 11):
        s = f"{sid:02d}"
        oi, oe = intra[s]["original"], inter[s]["original"]
        a, b = float(oi["top1"]), float(oe["top1"])
        t1i.append(a)
        t1e.append(b)
        print(
            f" {s}   {a:8.1f}  {float(oi['top5']):8.1f}   {b:8.1f}  {float(oe['top5']):8.1f}   {a - b:+7.1f}"
        )
    print(
        f" mean   {st.mean(t1i):8.1f}            {st.mean(t1e):8.1f}            {st.mean(t1i) - st.mean(t1e):+7.1f}"
    )


if __name__ == "__main__":
    main()
