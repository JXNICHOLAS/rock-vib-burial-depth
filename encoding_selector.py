#!/usr/bin/env python3
"""
encoding_selector.py
====================
Nested input-encoding comparison assembler (paper Table V, raw-pairs row).

For each outer LORO fold, the encoding whose inner-selected configuration
achieved the lower inner cross-validation MAPE is recorded, and that
encoding's outer predictions are stitched for the held-out rock. Because
both encodings' inner scores and outer predictions are produced by
nested_cv.py under an identical inner protocol, this analysis is
assembled from those artifacts without retraining.

Requires (run first):
  python nested_cv.py                      # meandiff11 (paper model)
  python nested_cv.py --encoding raw11

Outputs (to ./output/):
  encoding_selection.csv           — per-fold winner and inner-score margin
  nested_preds_allseeds_encsel.csv — stitched outer predictions

Paper context: under the nested protocol the mean/difference encoding
reaches 25.1 +/- 1.0 % versus 27.1 +/- 0.9 % for raw pairs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"


def main():
    sel, preds = {}, {}
    for enc in ("meandiff11", "raw11"):
        s = OUT / f"nested_selected_configs_{enc}.csv"
        p = OUT / f"nested_preds_allseeds_{enc}.csv"
        if not (s.exists() and p.exists()):
            raise SystemExit(f"missing {s.name} or {p.name} — run "
                             f"nested_cv.py --encoding {enc} first")
        sel[enc] = pd.read_csv(s).set_index("rock_out")
        preds[enc] = pd.read_csv(p)

    rows, parts = [], []
    for rock in sel["meandiff11"].index:
        scores = {enc: float(sel[enc].loc[rock, "inner_mape"]) for enc in sel}
        winner = min(scores, key=scores.get)
        rows.append(dict(rock_out=rock, winner=winner,
                         meandiff11_inner=round(scores["meandiff11"], 2),
                         raw11_inner=round(scores["raw11"], 2),
                         margin=round(abs(scores["meandiff11"] - scores["raw11"]), 2)))
        parts.append(preds[winner][preds[winner]["rock"] == rock]
                     .assign(encoding=winner))

    table = pd.DataFrame(rows)
    joint = pd.concat(parts, ignore_index=True)
    table.to_csv(OUT / "encoding_selection.csv", index=False)
    joint.to_csv(OUT / "nested_preds_allseeds_encsel.csv", index=False)

    per = joint.groupby("seed").apply(lambda g: pd.Series(
        {"mape": float(np.mean(np.abs(g.h_pred - g.h_true) / g.h_true) * 100),
         "rmse": float(np.sqrt(np.mean((g.h_pred - g.h_true) ** 2)))}))
    print(table.to_string(index=False))
    print(f"\nwins: {table.winner.value_counts().to_dict()}")
    print(f"encoding-selector result: "
          f"{per['mape'].mean():.1f} +/- {per['mape'].std(ddof=0):.1f} % / "
          f"{per['rmse'].mean():.3f} +/- {per['rmse'].std(ddof=0):.3f} cm")


if __name__ == "__main__":
    main()
