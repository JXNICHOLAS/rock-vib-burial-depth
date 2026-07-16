"""
nn_single_dir.py
================
Ablation: trains on single-direction measurements only (no pairing, no
mean/diff encoding) and compares against the best paired model.

Single-direction features (6):
    fn, Mp_raw, damping_ratio, beta_raw, w_mean, he

Both b and s measurements are used as independent training samples
(the model never sees b and s together for the same rock/condition).

This quantifies how much the directional pairing — and the diff features
that capture b vs s asymmetry — contribute to performance.

LORO CV: hold out all measurements (both b and s) of each physical rock.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "nn_results"
CSV    = BASE / "training_dataset.csv"

# ═════════════════════════════════════════════════════════════════════════════
# TUNABLE SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
HIDDEN_LAYERS    = (16, 8)
ACTIVATION       = "tanh"
ALPHA            = 5.0
RANDOM_SEED      = 0
MAX_ITER         = 50_000
LEARNING_RATE    = 5e-4
N_ITER_NO_CHANGE = 300
TOL              = 1e-6
TARGET           = "h_cm"
# ═════════════════════════════════════════════════════════════════════════════

OUTDIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load raw data
# ─────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(CSV)
raw = raw.dropna(subset=["fn_peak_Hz", "fn_halfpower_Hz", "damping_ratio", "npz_file",
                          "axis1_cm", "axis2_cm",
                          "h_cm", "Hs_cm", "M_kg", "direction", "burial_pct",
                          "log_spatial_slope"])
raw["fn_avg_Hz"] = (raw["fn_peak_Hz"] + raw["fn_halfpower_Hz"]) / 2

def load_peak_mag(npz_path, fd_hz):
    try:
        d   = np.load(npz_path)
        idx = np.argmin(np.abs(d["frequency_Hz"] - fd_hz))
        return float(d["magnitude"][idx])
    except Exception:
        return np.nan

print("Loading peak magnitudes from NPZ files ...", flush=True)
raw["peak_mag"] = [load_peak_mag(r["npz_file"], r["fn_peak_Hz"]) for _, r in raw.iterrows()]
raw = raw.dropna(subset=["peak_mag"])
print(f"  {len(raw)} rows retained.\n", flush=True)

raw["w_mean"] = (raw["axis1_cm"] + raw["axis2_cm"]) / 2
raw["w_diff"] = raw["axis2_cm"] - raw["axis1_cm"]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build paired dataset  (for baseline reference)
# ─────────────────────────────────────────────────────────────────────────────
raw["orient"] = raw["date"].str.split("_").str[-1]
pair_rows = []
for (rock, pct, orient), grp in raw.groupby(["rock", "burial_pct", "orient"]):
    b = grp[grp["direction"] == "b"]
    s = grp[grp["direction"] == "s"]
    if len(b) == 0 or len(s) == 0:
        continue
    ax1 = grp["axis1_cm"].iloc[0];  ax2 = grp["axis2_cm"].iloc[0]
    Hs  = grp["Hs_cm"].iloc[0];     h   = grp["h_cm"].iloc[0]
    fn_b = b["fn_avg_Hz"].mean();    fn_s = s["fn_avg_Hz"].mean()
    Mp_b = b["peak_mag"].mean();     Mp_s = s["peak_mag"].mean()
    zb   = b["damping_ratio"].mean(); zs  = s["damping_ratio"].mean()
    beta_b = b["log_spatial_slope"].mean(); beta_s = s["log_spatial_slope"].mean()
    pair_rows.append(dict(
        rock=rock, pct=pct,
        w_mean=(ax1+ax2)/2, w_diff=ax2-ax1, he=Hs,
        fn_mean=(fn_b+fn_s)/2, fn_diff=fn_b-fn_s,
        Mp_mean=(Mp_b+Mp_s)/2, Mp_diff=Mp_b-Mp_s,
        zeta_mean=(zb+zs)/2, zeta_diff=zb-zs,
        beta_mean=(beta_b+beta_s)/2, beta_diff=beta_b-beta_s,
        h_cm=h,
    ))

df_paired = pd.DataFrame(pair_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Single-direction dataset  — each row is one measurement
# ─────────────────────────────────────────────────────────────────────────────
FEAT_SINGLE = ["fn_avg_Hz", "peak_mag", "damping_ratio", "log_spatial_slope",
               "w_mean", "w_diff", "Hs_cm"]
df_single = raw[["rock", "burial_pct", "direction"] + FEAT_SINGLE + ["h_cm"]].copy()
df_single = df_single.rename(columns={"Hs_cm": "he"})
FEAT_SINGLE[-1] = "he"   # update column name

print(f"Paired  dataset : {len(df_paired)} rows   Rocks: {df_paired['rock'].nunique()}")
print(f"Single  dataset : {len(df_single)} rows   Rocks: {df_single['rock'].nunique()}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LORO helpers
# ─────────────────────────────────────────────────────────────────────────────
DIFF_COLS = ["w_diff","fn_diff","Mp_diff","zeta_diff","beta_diff"]
FEAT_PAIRED = ["w_mean","fn_mean","Mp_mean","he",
               "w_diff","fn_diff","Mp_diff",
               "zeta_mean","zeta_diff","beta_mean","beta_diff"]

def make_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION,
            alpha=ALPHA, max_iter=MAX_ITER,
            learning_rate_init=LEARNING_RATE,
            n_iter_no_change=N_ITER_NO_CHANGE, tol=TOL,
            random_state=RANDOM_SEED, solver="adam",
        ))
    ])


def paired_loro(df, feats):
    """LORO on paired dataset with b/s swap augmentation."""
    rocks = df["rock"].unique()
    y_true, y_pred = [], []
    for rock in rocks:
        tr_orig = df[df["rock"] != rock]
        tr_swap = tr_orig.copy()
        for col in DIFF_COLS:
            if col in tr_swap.columns:
                tr_swap[col] = -tr_orig[col]
        tr = pd.concat([tr_orig, tr_swap], ignore_index=True)
        te = df[df["rock"] == rock]
        pipe = make_pipe()
        pipe.fit(tr[feats].values, tr[TARGET].values)
        y_true.extend(te[TARGET].tolist())
        y_pred.extend(pipe.predict(te[feats].values).tolist())
    return np.array(y_true), np.array(y_pred)


def single_loro(df, feats):
    """LORO on single-direction dataset (no augmentation needed —
    both b and s are already present as separate rows)."""
    rocks = df["rock"].unique()
    y_true, y_pred = [], []
    for rock in rocks:
        tr = df[df["rock"] != rock]
        te = df[df["rock"] == rock]
        pipe = make_pipe()
        pipe.fit(tr[feats].values, tr[TARGET].values)
        y_true.extend(te[TARGET].tolist())
        y_pred.extend(pipe.predict(te[feats].values).tolist())
    return np.array(y_true), np.array(y_pred)


def report(label, yt, yp):
    mape = np.mean(np.abs(yt - yp) / np.abs(yt)) * 100
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    print(f"  {label:<38}  MAPE={mape:5.1f}%   RMSE={rmse:.3f} cm")
    return mape, rmse


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SINGLE DIRECTION vs PAIRED  --  LORO Cross-Validation")
print("=" * 65)
print(f"  Paired  features ({len(FEAT_PAIRED)}): {FEAT_PAIRED}")
print(f"  Single  features ({len(FEAT_SINGLE)}): {FEAT_SINGLE}")
print()

results = {}

print("Running Paired (11-input mean/diff, current best) ...")
yt, yp = paired_loro(df_paired, FEAT_PAIRED)
results["Paired\n(11-input mean/diff)"] = (*report("Paired  11-input mean/diff", yt, yp), yt, yp)

print("Running Single-direction (6 features, b and s as separate rows) ...")
yt, yp = single_loro(df_single, FEAT_SINGLE)
results["Single direction\n(6 features, b+s separate)"] = (*report("Single  6-input", yt, yp), yt, yp)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  {'Model':<38}  {'MAPE':>7}  {'RMSE':>9}  {'n_train':>8}")
print(f"  {'-'*38}  {'-'*7}  {'-'*9}  {'-'*8}")
n_paired = len(df_paired)
n_single = len(df_single)
for (name, (mape, rmse, _, __)), n in zip(results.items(), [n_paired, n_single]):
    label = name.replace("\n", " ")
    print(f"  {label:<38}  {mape:>6.1f}%  {rmse:>7.3f} cm  {n:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle("Ablation: Paired (b+s mean/diff) vs Single-Direction\n"
             "LORO CV  |  same MLP (8,4) tanh  |  alpha=1.0  |  7 vs 11 features",
             fontsize=11, fontweight="bold")

for ax, (label, (mape, rmse, yt, yp)) in zip(axes, results.items()):
    lim = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.4, s=45)
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="perfect")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True h (cm)", fontsize=10)
    ax.set_ylabel("Predicted h (cm)", fontsize=10)
    ax.set_title(f"{label}\nMAPE={mape:.1f}%   RMSE={rmse:.3f} cm", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
out_png = OUTDIR / "nn_single_dir_comparison.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"\n  Plot saved -> {out_png.name}")
