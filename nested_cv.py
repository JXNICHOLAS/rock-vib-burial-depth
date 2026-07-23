#!/usr/bin/env python3
"""
nested_cv.py
============
Nested Leave-One-Rock-Out cross-validation for the burial-depth estimator
(paper Section VI-C). This is the protocol behind the paper's headline
result: 25.1 +/- 1.0 % MAPE / 0.94 +/- 0.03 cm RMSE for the eleven-feature
mean/difference encoding (raw-pairs alternative: 27.1 +/- 0.9 %).

Hyperparameters are selected INSIDE each training fold so the held-out
rock never influences selection (Varma & Simon 2006; Cawley & Talbot 2010):

  OUTER: strict LORO (18 folds; all measurements of one rock held out).
  INNER: 5-fold cross-validation grouped by rock on the 17 training rocks,
         over a 96-configuration grid concentrated on small tanh networks
         by preliminary training-fold experiments (paper Sec. VI-C).
  The inner-selected configuration is refit on all training rocks and
  predicts the held-out rock; outer predictions are averaged over seeds.

Usage:
  python nested_cv.py                          # meandiff11 (paper model)
  python nested_cv.py --encoding raw11         # raw-pairs comparison row
  python nested_cv.py --encoding meandiff9     # damping-only candidate
  python nested_cv.py --encoding meandiff_beta9  # spatial-only candidate
  python nested_cv.py --deploy                 # inner rule on ALL 18 rocks
                                               # -> named deployed config
  (analysis/feature_set_selector.py assembles the paper's feature-set selection
   analysis from the three meandiff runs; results/ ships canonical outputs)

Outputs (to ./output/):
  nested_selected_configs_<enc>.csv  — per-fold selected configuration
  nested_preds_allseeds_<enc>.csv    — per-sample predictions, seeds 0-19
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
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paired_dataset import (load_paired, FEAT_RAW9, FEAT_RAW11,
                            FEAT_MEANDIFF9, FEAT_MEANDIFF11,
                            FEAT_MEANDIFF_B9, TARGET)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = Path(__file__).resolve().parent

# meandiff9 / meandiff_beta9 / meandiff11 are the three candidate feature
# sets of the paper's feature-set selection analysis (assembled by
# analysis/feature_set_selector.py); meandiff11 vs raw11 is the paper's input
# encoding comparison. raw9 is retained for completeness.
ENCODINGS = {"raw9": FEAT_RAW9, "raw11": FEAT_RAW11,
             "meandiff9": FEAT_MEANDIFF9, "meandiff11": FEAT_MEANDIFF11,
             "meandiff_beta9": FEAT_MEANDIFF_B9}

HIDDEN = [(8,), (12,),
          (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (16, 8), (20, 10), (24, 12),
          (8, 8), (16, 16),
          (8, 4, 2), (12, 6, 3), (16, 8, 4), (24, 12, 6)]
ACTS = ["tanh"]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
GRID = list(itertools.product(HIDDEN, ACTS, ALPHAS))
SEL_SEED = 0
OUTER_SEEDS = range(20)


def make_model(hidden, act, alpha, seed):
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=hidden, activation=act,
                             alpha=alpha, random_state=seed, max_iter=50_000,
                             learning_rate_init=5e-4, n_iter_no_change=300,
                             tol=1e-6)),
    ])


def mape(yt, yp):
    return float(np.mean(np.abs(yt - yp) / yt) * 100)


def inner_score(train_df, features, hidden, act, alpha):
    X = train_df[features].values
    y = train_df[TARGET].values
    groups = train_df["rock"].values
    errs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_model(hidden, act, alpha, SEL_SEED)
        m.fit(X[tr], y[tr])
        errs.append(mape(y[va], m.predict(X[va])))
    return float(np.mean(errs))


def select_fold(df, features, rock_out):
    train = df[df["rock"] != rock_out]
    scores = [inner_score(train, features, h, a, al) for (h, a, al) in GRID]
    i = int(np.argmin(scores))
    return rock_out, GRID[i], float(scores[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoding", choices=sorted(ENCODINGS), default="meandiff11")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--deploy", action="store_true",
                   help="apply the inner-selection rule to ALL 18 rocks and "
                        "print the deployed configuration (no outer eval)")
    args = p.parse_args()
    features = ENCODINGS[args.encoding]

    if args.deploy:
        df = load_paired()
        scores = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(inner_score)(df, features, h, a, al) for (h, a, al) in GRID)
        order = np.argsort(scores)
        print(f"\nDeployed configuration ({args.encoding}), inner rule on all "
              f"{df['rock'].nunique()} rocks:")
        for i in order[:5]:
            h, a, al = GRID[i]
            print(f"  {str(h):>12} {a} alpha={al:<4}  inner={scores[i]:.1f}%")
        return

    out = BASE / "output"
    out.mkdir(exist_ok=True)
    ckpt = out / f"nested_selected_configs_{args.encoding}.csv"

    df = load_paired()
    rocks = list(df["rock"].unique())
    t0 = time.time()

    sel, sc = {}, {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        for _, r in prev.iterrows():
            sel[r["rock_out"]] = (eval(r["hidden"]), r["act"], float(r["alpha"]))
            sc[r["rock_out"]] = float(r["inner_mape"])
        print(f"Resuming: {len(sel)}/{len(rocks)} folds already selected")

    todo = [r for r in rocks if r not in sel]
    if todo:
        print(f"Inner selection: {len(todo)} folds x {len(GRID)} configs "
              f"x 5 inner folds ...", flush=True)
        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(select_fold)(df, features, r) for r in todo)
        for (r, cfg, s) in results:
            sel[r] = cfg
            sc[r] = s
        pd.DataFrame([{"rock_out": r, "hidden": str(sel[r][0]),
                       "act": sel[r][1], "alpha": sel[r][2],
                       "inner_mape": sc[r]} for r in sel]).to_csv(ckpt, index=False)
    print(f"Selection done in {(time.time() - t0)/60:.1f} min")
    for r in rocks:
        print(f"  hold-out {r:>4}: {str(sel[r][0]):>10} {sel[r][1]:>4} "
              f"a={sel[r][2]:<4}  inner={sc[r]:.1f}%")

    def outer_seed(seed):
        rows = []
        for r in rocks:
            h, a, al = sel[r]
            train = df[df["rock"] != r]
            test = df[df["rock"] == r]
            m = make_model(h, a, al, seed)
            m.fit(train[features].values, train[TARGET].values)
            yp = m.predict(test[features].values)
            for (_, row), pred in zip(test.iterrows(), yp):
                rows.append(dict(rock=row["rock"], pct=row["pct"],
                                 date=row["date"], h_true=row["h_cm"],
                                 h_pred=float(pred), seed=seed))
        return pd.DataFrame(rows)

    print("Outer prediction (seeds 0-19) ...", flush=True)
    preds = pd.concat(Parallel(n_jobs=args.jobs)(
        delayed(outer_seed)(s) for s in OUTER_SEEDS), ignore_index=True)
    preds.to_csv(out / f"nested_preds_allseeds_{args.encoding}.csv", index=False)

    per_seed = preds.groupby("seed").apply(
        lambda g: pd.Series({"mape": mape(g.h_true.values, g.h_pred.values),
                             "rmse": float(np.sqrt(np.mean((g.h_pred - g.h_true) ** 2)))}))
    print("\n" + "=" * 60)
    print(f"NESTED CV RESULT ({args.encoding})")
    print("=" * 60)
    print(f"  h MAPE : {per_seed['mape'].mean():.1f}% +/- {per_seed['mape'].std(ddof=0):.1f}")
    print(f"  h RMSE : {per_seed['rmse'].mean():.3f} +/- {per_seed['rmse'].std(ddof=0):.3f} cm")
    print(f"  total time {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
