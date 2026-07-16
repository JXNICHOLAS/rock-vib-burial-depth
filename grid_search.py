#!/usr/bin/env python3
"""
grid_search.py
==============
Flat grid search over MLP hyperparameters, scored by mean LORO error
(90 configurations: 9 hidden-layer sizes x 2 activations x 5 alphas).

This is the selection procedure that originally identified the fixed
reference configuration (16,8)/tanh/alpha=5 reported in the paper.

IMPORTANT CAVEAT (paper Section VI): because this search scores each
configuration on the same LORO folds used for evaluation, using the
winning configuration's score as a generalization estimate is
optimistically biased (Varma & Simon 2006; Cawley & Talbot 2010). For the
paper's dataset the bias was measured at +3.1 percentage points MAPE.
Use nested_cv.py for an unbiased estimate; this script is provided for
transparency and reproducibility of the selection step only.

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

from paired_dataset import load_paired, FEAT_MEANDIFF, TARGET

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = Path(__file__).resolve().parent

HIDDEN = [(4,), (8,), (16,), (4, 2), (8, 4), (16, 8), (32, 16), (8, 4, 2), (16, 8, 4)]
ACTS = ["tanh", "relu"]
ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]
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
        m.fit(tr[FEAT_MEANDIFF].values, tr[TARGET].values)
        yt.extend(te[TARGET].values)
        yp.extend(m.predict(te[FEAT_MEANDIFF].values))
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
    print(f"Grid: {len(GRID)} configs x {args.seeds} seeds x 16 LORO folds")
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
