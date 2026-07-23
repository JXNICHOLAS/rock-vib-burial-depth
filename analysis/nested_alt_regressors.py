#!/usr/bin/env python3
"""
nested_alt_regressors.py
========================
Alternative regressor families under the SAME nested LORO protocol as the
proposed MLP (paper Table V, nested block), each with a pre-declared
hyperparameter grid selected per fold by rock-grouped 5-fold inner CV:

  GP : 3 kernel structures (ARD-RBF+White, iso-RBF+White, ARD-Matern1.5+White)
  RF : depth {None,4,8} x min_leaf {1,2,4} x max_features {1.0,'sqrt'}  (18)
  GB : n_est {100,300} x depth {1,2,3} x lr {0.03,0.1} x subsample {0.8,1.0} (24)

Paper results (eleven mean/difference features, 18 rocks): GP 39.6%,
RF 45.2%, GB 40.6% MAPE (vs MLP 25.1% / 0.94 cm).
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from paired_dataset import load_paired, FEAT_MEANDIFF11, TARGET

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]  # repo root
NFEAT = len(FEAT_MEANDIFF11)
SEL_SEED = 0
OUTER_SEEDS = range(20)


def gp_maker(kind):
    def make(seed):
        if kind == "ard_rbf":
            k = ConstantKernel(1.0) * RBF(np.ones(NFEAT)) + WhiteKernel(0.1)
        elif kind == "iso_rbf":
            k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        else:
            k = ConstantKernel(1.0) * Matern(np.ones(NFEAT), nu=1.5) + WhiteKernel(0.1)
        return Pipeline([("sc", StandardScaler()),
                         ("gp", GaussianProcessRegressor(
                             kernel=k, alpha=1e-6, normalize_y=True,
                             n_restarts_optimizer=2, random_state=seed))])
    return make


def rf_maker(depth, leaf, mf):
    def make(seed):
        return RandomForestRegressor(n_estimators=400, max_depth=depth,
                                     min_samples_leaf=leaf, max_features=mf,
                                     random_state=seed, n_jobs=1)
    return make


def gb_maker(n, depth, lr, sub):
    def make(seed):
        return GradientBoostingRegressor(n_estimators=n, max_depth=depth,
                                         learning_rate=lr, subsample=sub,
                                         random_state=seed)
    return make


GRIDS = {
    "GP": [(f"gp:{k}", gp_maker(k))
           for k in ("ard_rbf", "iso_rbf", "ard_matern15")],
    "RF": [(f"rf:d{d}_l{l}_m{m}", rf_maker(d, l, m))
           for d in (None, 4, 8) for l in (1, 2, 4) for m in (1.0, "sqrt")],
    "GB": [(f"gb:n{n}_d{d}_lr{lr}_s{s}", gb_maker(n, d, lr, s))
           for n in (100, 300) for d in (1, 2, 3) for lr in (0.03, 0.1)
           for s in (0.8, 1.0)],
}


def mape(yt, yp):
    return float(np.mean(np.abs(yt - yp) / yt) * 100)


def inner_score(train_df, make):
    X = train_df[FEAT_MEANDIFF11].values
    y = train_df[TARGET].values
    g = train_df["rock"].values
    errs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        m = make(SEL_SEED)
        m.fit(X[tr], y[tr])
        errs.append(mape(y[va], m.predict(X[va])))
    return float(np.mean(errs))


def select_fold(df, grid, rock_out):
    tr = df[df["rock"] != rock_out]
    scores = [inner_score(tr, make) for (_, make) in grid]
    i = int(np.argmin(scores))
    return rock_out, i, grid[i][0]


def run_model(df, name, grid, jobs):
    rocks = list(df["rock"].unique())
    t0 = time.time()
    sels = Parallel(n_jobs=jobs)(
        delayed(select_fold)(df, grid, r) for r in rocks)
    sel = {r: grid[i][1] for (r, i, tag) in sels}
    tags = {r: tag for (r, i, tag) in sels}

    def outer_seed(seed):
        yt, yp = [], []
        for r in rocks:
            tr = df[df["rock"] != r]
            te = df[df["rock"] == r]
            m = sel[r](seed)
            m.fit(tr[FEAT_MEANDIFF11].values, tr[TARGET].values)
            yt.extend(te[TARGET].values)
            yp.extend(m.predict(te[FEAT_MEANDIFF11].values))
        yt = np.array(yt)
        yp = np.array(yp)
        return mape(yt, yp), float(np.sqrt(np.mean((yt - yp) ** 2)))

    outs = Parallel(n_jobs=jobs)(delayed(outer_seed)(s) for s in OUTER_SEEDS)
    ms = [o[0] for o in outs]
    rs = [o[1] for o in outs]
    print(f"\n[{name}] nested: MAPE {np.mean(ms):.1f} +/- {np.std(ms):.1f}%   "
          f"RMSE {np.mean(rs):.3f} +/- {np.std(rs):.3f} cm   "
          f"({(time.time() - t0)/60:.1f} min)")
    for tag, n in pd.Series(list(tags.values())).value_counts().items():
        print(f"    {n:2d}/{len(rocks)}  {tag}")
    return dict(model=name, mape=np.mean(ms), mape_std=np.std(ms),
                rmse=np.mean(rs), rmse_std=np.std(rs))


def main():
    df = load_paired()
    out = BASE / "output"
    out.mkdir(exist_ok=True)
    rows = [run_model(df, name, grid, jobs=-1) for name, grid in GRIDS.items()]
    pd.DataFrame(rows).to_csv(out / "nested_alt_regressors.csv", index=False)
    print("\nReference: nested MLP (meandiff11) = 25.1 +/- 1.0 % / 0.94 cm")


if __name__ == "__main__":
    main()
