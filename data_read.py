import csv, pickle
from pathlib import Path

pkl_path = Path("test_pt/synthesisStatistics.pickle")
out_csv = Path("test_pt/synthesisStatistics_long.csv")

with pkl_path.open("rb") as f:
    stats = pickle.load(f)  # dict: design -> [ANDgates, NOTgates, lpLen, area, delay]

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Design", "RecipeID", "ANDgates", "NOTgates", "lpLen", "area", "delay"])
    for design, arrs in stats.items():
        ANDg, NOTg, lp, area, delay = arrs
        n = min(len(ANDg), len(NOTg), len(lp), len(area), len(delay))
        for sid in range(n):
            w.writerow([design, sid, ANDg[sid], NOTg[sid], lp[sid], area[sid], delay[sid]])

print("wrote:", out_csv)

