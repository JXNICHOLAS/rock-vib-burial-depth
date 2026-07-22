#!/usr/bin/env python3
"""
nn_single_dir.py
================
Single-direction benchmark (paper Table II, last row): how much accuracy is
lost when only ONE of the two orthogonal measurements is used per burial
state. Each individual measurement becomes one sample with seven features

    [w_b, w_s, he, fn, Mp, zeta, beta]

(the geometric widths are known regardless of which face is struck; the
four FRF quantities come from the single measurement, matching the
proposed eleven-feature set's inclusion of beta).

Protocol: strict LORO at the fixed reference configuration
(16, 8, 4)/tanh/alpha=5, seeds 0-19 (matched relative comparison; see
nested_cv.py for the absolute protocol). Both directions contribute
samples ("one measurement per inference, direction not chosen").

Paper result: 29.6 +/- 0.6 % / 1.28 cm (versus 22.6 +/- 0.8 % for the
paired eleven-feature model at the same fixed configuration).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from paired_dataset import widths, _load_peak_mag, _beta_ls  # noqa: E402

CSV = BASE / "training_dataset.csv"
FEAT_SINGLE = ["w_b", "w_s", "he", "fn", "Mp", "zeta", "beta"]
TARGET = "h_cm"
FIXED = dict(hidden_layer_sizes=(16, 8, 4), activation="tanh", alpha=5.0)
SEEDS = range(20)


def load_single():
    raw = pd.read_csv(CSV)
    raw = raw.dropna(subset=["fn_peak_Hz", "fn_halfpower_Hz", "damping_ratio",
                             "npz_file", "axis1_cm", "axis2_cm", "axis3_cm",
                             "h_cm", "Hs_cm", "direction", "burial_pct"])
    raw["fn"] = (raw["fn_peak_Hz"] + raw["fn_halfpower_Hz"]) / 2
    raw["Mp"] = [_load_peak_mag(r["npz_file"], r["fn_peak_Hz"])
                 for _, r in raw.iterrows()]
    raw["beta"] = [_beta_ls(r["npz_file"]) for _, r in raw.iterrows()]
    raw = raw.dropna(subset=["Mp", "beta"])
    raw["orient"] = raw["date"].str.split("_").str[-1]
    ws_wb = raw.apply(lambda r: widths(r["orient"], r["axis1_cm"],
                                       r["axis2_cm"], r["axis3_cm"]), axis=1)
    raw["w_s"] = [t[0] for t in ws_wb]
    raw["w_b"] = [t[1] for t in ws_wb]
    raw["he"] = raw["Hs_cm"]
    raw["zeta"] = raw["damping_ratio"]
    return raw[["rock", "burial_pct", "orient", "direction"]
               + FEAT_SINGLE + [TARGET]].reset_index(drop=True)


def make_model(seed):
    return Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPRegressor(random_state=seed, max_iter=50_000,
                                          learning_rate_init=5e-4,
                                          n_iter_no_change=300, tol=1e-6,
                                          **FIXED))])


def loro(df, seed):
    yt, yp = [], []
    for r in df["rock"].unique():
        tr = df[df["rock"] != r]
        te = df[df["rock"] == r]
        m = make_model(seed)
        m.fit(tr[FEAT_SINGLE].values, tr[TARGET].values)
        yt.extend(te[TARGET].values)
        yp.extend(m.predict(te[FEAT_SINGLE].values))
    yt, yp = np.array(yt), np.array(yp)
    return (float(np.mean(np.abs(yt - yp) / yt) * 100),
            float(np.sqrt(np.mean((yt - yp) ** 2))))


def main():
    df = load_single()
    print(f"{len(df)} single-direction samples, {df['rock'].nunique()} rocks")
    outs = Parallel(n_jobs=-1)(delayed(loro)(df, s) for s in SEEDS)
    m = [o[0] for o in outs]
    r = [o[1] for o in outs]
    print(f"single-direction ({len(FEAT_SINGLE)} features, "
          f"fixed config): MAPE {np.mean(m):.1f} +/- {np.std(m):.1f} %   "
          f"RMSE {np.mean(r):.3f} +/- {np.std(r):.3f} cm")


if __name__ == "__main__":
    main()
