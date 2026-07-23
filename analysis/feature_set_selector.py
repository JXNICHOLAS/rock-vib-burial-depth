#!/usr/bin/env python3
"""
feature_set_selector.py
=======================
Nested feature-set selection (paper Sec. VII-D, "Feature-set selection"),
conducted under the mean/difference encoding: for each outer LORO fold the
inner loop compares the damping set (meandiff9), the spatial-slope set
(meandiff_beta9), and their combination (meandiff11) — 3 x 96 = 288
candidates per fold — and the winning set's outer predictions are used for
the held-out rock. Assembled from nested_cv.py artifacts without
retraining, exactly like encoding_selector.py.

Requires (run first):
  python nested_cv.py --encoding meandiff9
  python nested_cv.py --encoding meandiff_beta9
  python nested_cv.py --encoding meandiff11

Outputs (to the repo-root output/):
  featureset_selection.csv           — per-fold winner and margins
  nested_preds_allseeds_fsel.csv     — stitched outer predictions

Paper result: the combined eleven-feature set is selected in 10 of 18
outer folds and the damping-only set in 8; the beta-only set is never
selected. The adaptive procedure reaches 26.0 +/- 0.8 % / 0.964 cm.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # repo root
OUT = BASE / "output"
SETS = ("meandiff9", "meandiff_beta9", "meandiff11")


def main():
    sel, preds = {}, {}
    for enc in SETS:
        s = OUT / f"nested_selected_configs_{enc}.csv"
        p = OUT / f"nested_preds_allseeds_{enc}.csv"
        if not (s.exists() and p.exists()):
            raise SystemExit(f"missing {s.name} or {p.name} — run "
                             f"nested_cv.py --encoding {enc} first")
        sel[enc] = pd.read_csv(s).set_index("rock_out")
        preds[enc] = pd.read_csv(p)

    rows, parts = [], []
    for rock in sel[SETS[0]].index:
        scores = {enc: float(sel[enc].loc[rock, "inner_mape"]) for enc in SETS}
        winner = min(scores, key=scores.get)
        ordered = sorted(scores.values())
        rows.append(dict(rock_out=rock, winner=winner,
                         margin=round(ordered[1] - ordered[0], 2),
                         **{f"{enc}_inner": round(scores[enc], 2) for enc in SETS}))
        parts.append(preds[winner][preds[winner]["rock"] == rock]
                     .assign(feature_set=winner))

    table = pd.DataFrame(rows)
    joint = pd.concat(parts, ignore_index=True)
    table.to_csv(OUT / "featureset_selection.csv", index=False)
    joint.to_csv(OUT / "nested_preds_allseeds_fsel.csv", index=False)

    per = joint.groupby("seed").apply(lambda g: pd.Series(
        {"mape": float(np.mean(np.abs(g.h_pred - g.h_true) / g.h_true) * 100),
         "rmse": float(np.sqrt(np.mean((g.h_pred - g.h_true) ** 2)))}))
    print(table.to_string(index=False))
    print(f"\nwins: {table.winner.value_counts().to_dict()}")
    print(f"feature-set-selector result: "
          f"{per['mape'].mean():.1f} +/- {per['mape'].std(ddof=0):.1f} % / "
          f"{per['rmse'].mean():.3f} +/- {per['rmse'].std(ddof=0):.3f} cm")


if __name__ == "__main__":
    main()
