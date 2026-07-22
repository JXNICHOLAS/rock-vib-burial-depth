#!/usr/bin/env python3
"""
baseline_jia.py
===============
Known-mass physics baseline (paper Table V, calibrated physics-baseline row).

Evaluates the closed-form inversion of the Jia et al. (2022) spring--mass
model on the same strict LORO splits as the learned models, giving it every
advantage the data-driven estimator does NOT receive:
  - the TRUE rock mass m of each block (the MLP never observes mass), and
  - the soil-stiffness constant K = 4*pi^2/k fitted on the 17 training
    rocks of each fold (1-parameter fit minimizing training MAPE).

Jia model:  h/m = 4 pi^2 fn^2 (a^2 + he^2) / [(1 - zeta^2) k b a^2]
The geometric factor is averaged over both (a,b) cross-section orderings.

Paper result: 45.9% MAPE / 2.46 cm RMSE — nearly twice the error of the
mass-agnostic MLP (25.1%), because strain-dependent soil stiffness breaks
the assumed frequency-depth relationship (Section III).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from paired_dataset import load_paired

BASE = Path(__file__).resolve().parent


def horizontal_dims(row):
    """Two horizontal cross-section dims (cm) for this orientation."""
    if row["date"] == "z":            # ax3 vertical -> horizontal = ax1, ax2
        return row["axis1_cm"], row["axis2_cm"]
    return row["axis2_cm"], row["axis3_cm"]  # 'x': ax1 vertical


def jia_geom(row):
    """m * fn^2 * mean over (a,b) orderings of (a^2+he^2)/((1-z^2) b a^2)."""
    h1, h2 = horizontal_dims(row)
    he, z, f, m = row["he_cm"], row["zeta_mean"], row["fn_mean"], row["M_kg"]

    def term(a, b):
        return (a ** 2 + he ** 2) / ((1 - z ** 2) * b * a ** 2)

    return m * f ** 2 * 0.5 * (term(h1, h2) + term(h2, h1))


def fit_K(g, h):
    """1-parameter scan for K minimizing training MAPE (generous to Jia)."""
    k0 = np.median(h / g)
    grid = k0 * np.linspace(0.2, 3.0, 400)
    errs = [np.mean(np.abs(k * g - h) / h) for k in grid]
    return grid[int(np.argmin(errs))]


def main():
    df = load_paired().copy()
    df["g"] = df.apply(jia_geom, axis=1)

    yt, yp = [], []
    for r in df["rock"].unique():
        tr = df[df["rock"] != r]
        te = df[df["rock"] == r]
        K = fit_K(tr["g"].values, tr["h_cm"].values)
        yt.extend(te["h_cm"].values)
        yp.extend(np.clip(K * te["g"].values, 1e-6, None))
    yt = np.array(yt)
    yp = np.array(yp)

    print("=" * 60)
    print("JIA KNOWN-MASS BASELINE (strict LORO, K fit per training fold)")
    print("=" * 60)
    print(f"  h MAPE : {np.mean(np.abs(yt - yp) / yt) * 100:.1f}%")
    print(f"  h RMSE : {np.sqrt(np.mean((yt - yp) ** 2)):.3f} cm")
    print(f"  bias   : {np.mean(yp - yt):+.3f} cm")
    print(f"  worst  : {np.max(np.abs(yp - yt)):.2f} cm")

    out = BASE / "output"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"h_true": yt, "h_pred": yp}).to_csv(
        out / "baseline_jia_pred.csv", index=False)


if __name__ == "__main__":
    main()
