#!/usr/bin/env python3
"""
linear_baselines.py
===================
Deterministic baselines of paper Table V (LORO evaluation, no
hyperparameter selection), plus a training-mean dummy predictor for
context:

  linear regression        : 31.6 % MAPE / 1.29 cm
  polynomial regression d2 : 64.4 % MAPE / 2.48 cm
  training-mean dummy      : 58.3 % MAPE

All use the eleven mean/difference inputs of the proposed model (the
dummy ignores its inputs). Each model is refit per LORO fold on the 17
training rocks; no seeds are involved because every fit is deterministic.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from paired_dataset import load_paired, FEAT_MEANDIFF11, TARGET


def mape(t, p):
    return float(np.mean(np.abs(np.asarray(t) - np.asarray(p)) / np.asarray(t)) * 100)


def rmse(t, p):
    return float(np.sqrt(np.mean((np.asarray(t) - np.asarray(p)) ** 2)))


def loro(df, make):
    yt, yp = [], []
    for r in df["rock"].unique():
        tr, te = df[df.rock != r], df[df.rock == r]
        m = make()
        m.fit(tr[FEAT_MEANDIFF11].values, tr[TARGET].values)
        yt.extend(te[TARGET].values)
        yp.extend(m.predict(te[FEAT_MEANDIFF11].values))
    return mape(yt, yp), rmse(yt, yp)


def main():
    df = load_paired()
    for name, make in (
            ("linear", lambda: Pipeline([("sc", StandardScaler()),
                                         ("lr", LinearRegression())])),
            ("poly2", lambda: Pipeline([("sc", StandardScaler()),
                                        ("pf", PolynomialFeatures(2)),
                                        ("lr", LinearRegression())]))):
        m, r = loro(df, make)
        print(f"{name:7s}: MAPE {m:.1f} %   RMSE {r:.2f} cm")

    yt, yp = [], []
    for rock in df["rock"].unique():
        tr, te = df[df.rock != rock], df[df.rock == rock]
        yt.extend(te[TARGET].values)
        yp.extend([tr[TARGET].mean()] * len(te))
    print(f"dummy  : MAPE {mape(yt, yp):.1f} %  (training-mean predictor)")


if __name__ == "__main__":
    main()
