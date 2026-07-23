#!/usr/bin/env python3
"""
paired_dataset.py
=================
Shared loader for nested_cv.py and the analysis/ scripts (fixed-config
ablation, physics and linear baselines, alternative regressors, grid
search, and figure generation).

Builds the 102-pair dataset (one row per rock/burial/orientation) with:

  * Orientation-aware widths (paper Sec. IV-C): only the two horizontal,
    externally measurable dimensions are used. With block dims
    (axis1, axis2, axis3) = (x, y, z):
        z vertical -> (w_s, w_b) = (axis1, axis2)
        x vertical -> (w_s, w_b) = (axis2, axis3)
    The b/s labels match the two strike directions (larger/smaller face).
  * Spatial log-slope beta (paper Eq. 8): least-squares slope of
    ln(peak magnitude) vs. vertical scan coordinate over ALL valid scan
    points, computed from the per-point arrays stored in each NPZ file.
  * Both input encodings: the mean/difference representation (proposed
    model) and raw per-direction features (evaluated alternative).

Requires training_dataset.csv (run make_training_dataset.py first).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV = BASE / "training_dataset.csv"

# Raw per-direction encoding (evaluated alternative; paper Table V row).
FEAT_RAW9 = ["w_b", "w_s", "he", "fn_b", "fn_s", "Mp_b", "Mp_s",
             "zeta_b", "zeta_s"]
FEAT_RAW11 = FEAT_RAW9 + ["beta_b", "beta_s"]

# Proposed estimator: eleven mean/difference inputs (paper Table I),
# via FEAT_MEANDIFF11 below.
# NOTE: feature ORDER matters for exact reproduction — MLP weight init under a
# fixed seed maps to input columns by position, so this order matches the runs
# that produced the paper's numbers.
FEAT_MEANDIFF9 = ["w_mean", "fn_mean", "Mp_mean", "he",
                  "w_diff", "fn_diff", "Mp_diff", "zeta_mean", "zeta_diff"]
FEAT_MEANDIFF11 = FEAT_MEANDIFF9 + ["beta_mean", "beta_diff"]
FEAT_MEANDIFF_B9 = ["w_mean", "fn_mean", "Mp_mean", "he",
                    "w_diff", "fn_diff", "Mp_diff", "beta_mean", "beta_diff"]

TARGET = "h_cm"


def widths(orient, ax1, ax2, ax3):
    """(w_s, w_b): the two horizontal dims for this vertical orientation."""
    if orient == "x":                 # ax1 vertical -> horizontal = (ax2, ax3)
        return ax2, ax3
    return ax1, ax2                   # 'z': ax3 vertical -> horizontal = (ax1, ax2)


def _load_peak_mag(npz_path, fd_hz):
    try:
        d = np.load(npz_path)
        idx = np.argmin(np.abs(d["frequency_Hz"] - fd_hz))
        return float(d["magnitude"][idx])
    except Exception:
        return np.nan


def _beta_ls(npz_path):
    """All-point least-squares log-slope (paper Eq. 8), in 1/cm.

    The NPZ stores scan coordinates in millimeters; they are converted to
    centimeters before fitting so the slope matches Eq. 8's units. (The
    unit choice is absorbed by per-feature standardization, so it does not
    affect any model result.)"""
    try:
        d = np.load(npz_path)
        mags = np.asarray(d["point_mags_sorted"], dtype=float)
        ys = np.asarray(d["point_ymm_sorted"], dtype=float) / 10.0
        ok = (mags > 0) & np.isfinite(mags) & np.isfinite(ys)
        if ok.sum() < 3:
            return np.nan
        return float(np.polyfit(ys[ok], np.log(mags[ok]), 1)[0])
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
    raw["beta_ls"] = [_beta_ls(r["npz_file"]) for _, r in raw.iterrows()]
    raw = raw.dropna(subset=["peak_mag", "beta_ls"])
    raw["orient"] = raw["date"].str.split("_").str[-1]

    rows = []
    for (rock, pct, orient), grp in raw.groupby(["rock", "burial_pct", "orient"]):
        b = grp[grp["direction"] == "b"]
        s = grp[grp["direction"] == "s"]
        if len(b) == 0 or len(s) == 0:
            continue
        ax1 = grp["axis1_cm"].iloc[0]; ax2 = grp["axis2_cm"].iloc[0]; ax3 = grp["axis3_cm"].iloc[0]
        Hs = grp["Hs_cm"].iloc[0]; h = grp["h_cm"].iloc[0]; M = grp["M_kg"].iloc[0]
        w_s, w_b = widths(orient, ax1, ax2, ax3)
        fn_b = b["fn_avg_Hz"].mean(); fn_s = s["fn_avg_Hz"].mean()
        Mp_b = b["peak_mag"].mean(); Mp_s = s["peak_mag"].mean()
        zb = b["damping_ratio"].mean(); zs = s["damping_ratio"].mean()
        bb = b["beta_ls"].mean(); bs = s["beta_ls"].mean()
        rows.append(dict(
            rock=rock, pct=pct, date=orient,
            axis1_cm=ax1, axis2_cm=ax2, axis3_cm=ax3,
            he_cm=Hs, M_kg=M, density=(M * 1000.0) / (ax1 * ax2 * ax3),
            # raw per-direction encoding (evaluated alternative)
            w_b=w_b, w_s=w_s, he=Hs,
            fn_b=fn_b, fn_s=fn_s, Mp_b=Mp_b, Mp_s=Mp_s,
            zeta_b=zb, zeta_s=zs, beta_b=bb, beta_s=bs,
            # mean/difference representation (proposed model)
            w_mean=(w_b + w_s) / 2, w_diff=w_b - w_s,
            fn_mean=(fn_b + fn_s) / 2, fn_diff=fn_b - fn_s,
            Mp_mean=(Mp_b + Mp_s) / 2, Mp_diff=Mp_b - Mp_s,
            zeta_mean=(zb + zs) / 2, zeta_diff=zb - zs,
            beta_mean=(bb + bs) / 2, beta_diff=bb - bs,
            # horizontal cross-sectional area for derived-volume plots
            area_cm2=w_b * w_s,
            h_cm=h,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_paired()
    print(f"{len(df)} paired samples, {df['rock'].nunique()} rocks")
    print("proposed features:", FEAT_MEANDIFF11)
