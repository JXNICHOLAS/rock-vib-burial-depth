#!/usr/bin/env python3
"""
paired_dataset.py
=================
Shared loader for the revision-experiment scripts (nested_cv.py,
baseline_jia.py, nested_alt_regressors.py, error_characteristics.py,
grid_search.py).

Builds the same 90-pair dataset as nn_LORO.py (no augmentation), but
additionally retains per-direction values (fn_b, fn_s, ...) needed by the
raw-pairs encoding and the physics baseline.

Requires training_dataset.csv (run make_training_dataset.py first).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV = BASE / "training_dataset.csv"

FEAT_MEANDIFF = ["w_mean", "fn_mean", "Mp_mean", "he",
                 "w_diff", "fn_diff", "Mp_diff",
                 "zeta_mean", "zeta_diff", "beta_mean", "beta_diff"]

FEAT_RAWPAIRS = ["ax1", "ax2", "he", "fn_b", "fn_s", "Mp_b", "Mp_s",
                 "zeta_b", "zeta_s", "beta_b", "beta_s"]

TARGET = "h_cm"


def _load_peak_mag(npz_path, fd_hz):
    try:
        d = np.load(npz_path)
        idx = np.argmin(np.abs(d["frequency_Hz"] - fd_hz))
        return float(d["magnitude"][idx])
    except Exception:
        return np.nan


def load_paired(csv_path=CSV):
    """Return the paired DataFrame (one row per rock/burial/orientation)."""
    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["fn_peak_Hz", "fn_halfpower_Hz", "damping_ratio",
                             "npz_file", "axis1_cm", "axis2_cm", "axis3_cm",
                             "h_cm", "Hs_cm", "M_kg", "direction", "burial_pct",
                             "mag_gradient"])
    raw["fn_avg_Hz"] = (raw["fn_peak_Hz"] + raw["fn_halfpower_Hz"]) / 2
    raw["peak_mag"] = [_load_peak_mag(r["npz_file"], r["fn_peak_Hz"])
                       for _, r in raw.iterrows()]
    raw = raw.dropna(subset=["peak_mag", "log_spatial_slope"])
    raw["orient"] = raw["date"].str.split("_").str[-1]

    rows = []
    for (rock, pct, orient), grp in raw.groupby(["rock", "burial_pct", "orient"]):
        b = grp[grp["direction"] == "b"]
        s = grp[grp["direction"] == "s"]
        if len(b) == 0 or len(s) == 0:
            continue
        ax1 = grp["axis1_cm"].iloc[0]; ax2 = grp["axis2_cm"].iloc[0]; ax3 = grp["axis3_cm"].iloc[0]
        Hs = grp["Hs_cm"].iloc[0]; h = grp["h_cm"].iloc[0]; M = grp["M_kg"].iloc[0]
        fn_b = b["fn_avg_Hz"].mean(); fn_s = s["fn_avg_Hz"].mean()
        Mp_b = b["peak_mag"].mean(); Mp_s = s["peak_mag"].mean()
        zb = b["damping_ratio"].mean(); zs = s["damping_ratio"].mean()
        bb = b["log_spatial_slope"].mean(); bs = s["log_spatial_slope"].mean()
        rows.append(dict(
            rock=rock, pct=pct, date=orient,
            axis1_cm=ax1, axis2_cm=ax2, ax3_cm=ax3,
            he_cm=Hs, M_kg=M, density=(M * 1000.0) / (ax1 * ax2 * ax3),
            # mean/diff encoding
            w_mean=(ax1 + ax2) / 2, fn_mean=(fn_b + fn_s) / 2,
            Mp_mean=(Mp_b + Mp_s) / 2, he=Hs,
            w_diff=ax2 - ax1, fn_diff=fn_b - fn_s, Mp_diff=Mp_b - Mp_s,
            beta_mean=(bb + bs) / 2, beta_diff=bb - bs,
            zeta_mean=(zb + zs) / 2, zeta_diff=zb - zs,
            # raw-pairs encoding
            ax1_=ax1, ax2_=ax2,
            fn_b=fn_b, fn_s=fn_s, Mp_b=Mp_b, Mp_s=Mp_s,
            zeta_b=zb, zeta_s=zs, beta_b=bb, beta_s=bs,
            h_cm=h,
        ))
    df = pd.DataFrame(rows)
    df["ax1"] = df["ax1_"]; df["ax2"] = df["ax2_"]
    return df.drop(columns=["ax1_", "ax2_"])


if __name__ == "__main__":
    df = load_paired()
    print(f"{len(df)} paired samples, {df['rock'].nunique()} rocks")
