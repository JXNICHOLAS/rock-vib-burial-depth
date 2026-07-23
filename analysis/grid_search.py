#!/usr/bin/env python3
"""
grid_search.py
==============
Flat grid search over MLP hyperparameters, scored by mean LORO error
(96 configurations: 16 tanh hidden-layer shapes x 6 alphas — the refined
grid of paper Sec. VI-C, applied to the eleven-feature mean/difference encoding).

This is the style of preliminary screening that identified the fixed
reference configuration (16,8,4)/tanh/alpha=5 used for the paper's
matched ablation comparisons.

IMPORTANT CAVEAT (paper Section VI): because this search scores each
configuration on the same LORO folds used for evaluation, a winning
configuration's score is NOT a valid generalization estimate (Varma &
Simon 2006; Cawley & Talbot 2010). Absolute performance in the paper
comes from nested_cv.py, where selection is confined to training folds;
this script is provided for transparency of the screening step only.

Usage:
  python grid_search.py                 # 3 seeds (fast screening)
  python grid_search.py --seeds 20      # full 20-seed average
"""
import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from paired_dataset import load_paired, FEAT_MEANDIFF11, TARGET

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = Path(__file__).resolve().parents[1]  # repo root

HIDDEN = [(8,), (12,),
          (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (16, 8), (20, 10), (24, 12),
          (8, 8), (16, 16),
          (8, 4, 2), (12, 6, 3), (16, 8, 4), (24, 12, 6)]
ACTS = ["tanh"]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
GRID = list(itertools.product(HIDDEN, ACTS, ALPHAS))


def make_model(hidden, act, alpha, seed):
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=hidden, activation=act,
                             alpha=alpha, random_state=seed, max_iter=50_000,
                             learning_rate_init=5e-4, n_iter_no_change=300,
                             tol=1e-6)),
    ])


def loro_mape(df, hidden, act, alpha, seed):
    yt, yp = [], []
    for r in df["rock"].unique():
        tr = df[df["rock"] != r]
        te = df[df["rock"] == r]
        m = make_model(hidden, act, alpha, seed)
        m.fit(tr[FEAT_MEANDIFF11].values, tr[TARGET].values)
        yt.extend(te[TARGET].values)
        yp.extend(m.predict(te[FEAT_MEANDIFF11].values))
    yt = np.array(yt)
    yp = np.array(yp)
    return (float(np.mean(np.abs(yt - yp) / yt) * 100),
            float(np.sqrt(np.mean((yt - yp) ** 2))))


def eval_config(df, hidden, act, alpha, seeds):
    res = [loro_mape(df, hidden, act, alpha, s) for s in range(seeds)]
    return dict(hidden=str(hidden), activation=act, alpha=alpha,
                mape=float(np.mean([r[0] for r in res])),
                rmse=float(np.mean([r[1] for r in res])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()

    df = load_paired()
    print(f"Grid: {len(GRID)} configs x {args.seeds} seeds x 18 LORO folds")
    t0 = time.time()
    records = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(eval_config)(df, h, a, al, args.seeds) for (h, a, al) in GRID)
    res = pd.DataFrame(records).sort_values("mape").reset_index(drop=True)

    print(f"\nTop 10 (of {len(GRID)}), sorted by LORO MAPE "
          f"(SELECTION-BIASED — see docstring):")
    print(res.head(10).to_string(index=False))

    out = BASE / "output"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "grid_search_results.csv", index=False)
    print(f"\nSaved -> {out / 'grid_search_results.csv'}   "
          f"({(time.time() - t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
