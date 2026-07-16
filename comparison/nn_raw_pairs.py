"""
nn_raw_pairs.py
===============
Compares two input encodings for the same 11 features:

  A) Mean/Diff (current best):
       w_mean, fn_mean, Mp_mean, Hs, w_diff, fn_diff, Mp_diff,
       zeta_mean, zeta_diff, beta_mean, beta_diff

  B) Raw Pairs (this test):
       ax1, ax2, fn_b, fn_s, Mp_b, Mp_s, Hs,
       zb, zs, beta_b, beta_s

Both have exactly 11 inputs. The question is whether the NN benefits
from seeing fn_b and fn_s separately rather than as (fn_mean, fn_diff).

Augmentation:
  Mean/Diff  — negate all diff columns (b/s swap)
  Raw Pairs  — physically swap all _b / _s column pairs (same effect)
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
                          "axis1_cm", "axis2_cm", "axis3_cm",
                          "h_cm", "Hs_cm", "M_kg", "direction", "burial_pct",
                          "mag_gradient", "log_spatial_slope"])
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

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build paired dataset
# ─────────────────────────────────────────────────────────────────────────────
raw["orient"] = raw["date"].str.split("_").str[-1]
rows = []
for (rock, pct, orient), grp in raw.groupby(["rock", "burial_pct", "orient"]):
    b = grp[grp["direction"] == "b"]
    s = grp[grp["direction"] == "s"]
    if len(b) == 0 or len(s) == 0:
        continue

    ax1 = grp["axis1_cm"].iloc[0]
    ax2 = grp["axis2_cm"].iloc[0]
    Hs  = grp["Hs_cm"].iloc[0]
    h   = grp["h_cm"].iloc[0]

    fn_b = b["fn_avg_Hz"].mean();    fn_s = s["fn_avg_Hz"].mean()
    Mp_b = b["peak_mag"].mean();     Mp_s = s["peak_mag"].mean()
    zb   = b["damping_ratio"].mean(); zs  = s["damping_ratio"].mean()
    beta_b = b["log_spatial_slope"].mean(); beta_s = s["log_spatial_slope"].mean()

    rows.append(dict(
        rock=rock, pct=pct, date=orient,
        # Rock-level scalars
        ax1=ax1, ax2=ax2, he=Hs,
        # Mean/diff encoding
        w_mean=(ax1+ax2)/2,   w_diff=ax2-ax1,
        fn_mean=(fn_b+fn_s)/2, fn_diff=fn_b-fn_s,
        Mp_mean=(Mp_b+Mp_s)/2, Mp_diff=Mp_b-Mp_s,
        zeta_mean=(zb+zs)/2,   zeta_diff=zb-zs,
        beta_mean=(beta_b+beta_s)/2, beta_diff=beta_b-beta_s,
        # Raw per-direction
        fn_b=fn_b, fn_s=fn_s,
        Mp_b=Mp_b, Mp_s=Mp_s,
        zb=zb, zs=zs,
        beta_b=beta_b, beta_s=beta_s,
        h_cm=h,
    ))

df = pd.DataFrame(rows)
print(f"Pairs: {len(df)}   Rocks: {df['rock'].nunique()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature sets
# ─────────────────────────────────────────────────────────────────────────────
FEAT_MEANDIFF = ["w_mean", "fn_mean", "Mp_mean", "he",
                 "w_diff", "fn_diff", "Mp_diff",
                 "zeta_mean", "zeta_diff", "beta_mean", "beta_diff"]

FEAT_RAW      = ["ax1", "ax2", "he",
                 "fn_b", "fn_s",
                 "Mp_b", "Mp_s",
                 "zb",   "zs",
                 "beta_b", "beta_s"]

VARIANTS = [
    ("Mean/Diff (current best)", FEAT_MEANDIFF,
     ["w_diff","fn_diff","Mp_diff","zeta_diff","beta_diff"],   # cols to negate
     []),                                                       # raw pairs to swap
    ("Raw Pairs (fn_b, fn_s, …)", FEAT_RAW,
     [],                                                        # no diff cols
     [("fn_b","fn_s"), ("Mp_b","Mp_s"), ("zb","zs"),          # pairs to swap
      ("beta_b","beta_s"), ("ax1","ax2")]),
]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LORO runner
# ─────────────────────────────────────────────────────────────────────────────
def augment(df_in, diff_cols, swap_pairs):
    """Double dataset by swapping b/s."""
    swap = df_in.copy()
    for col in diff_cols:
        if col in swap.columns:
            swap[col] = -df_in[col]
    for ca, cb in swap_pairs:
        if ca in swap.columns and cb in swap.columns:
            swap[ca] = df_in[cb]
            swap[cb] = df_in[ca]
    return pd.concat([df_in, swap], ignore_index=True)


def run_loro(df, feats, diff_cols, swap_pairs):
    rocks = df["rock"].unique()
    loo_rows = []
    for rock in rocks:
        tr = augment(df[df["rock"] != rock], diff_cols, swap_pairs)
        te = df[df["rock"] == rock]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION,
                alpha=ALPHA, max_iter=MAX_ITER,
                learning_rate_init=LEARNING_RATE,
                n_iter_no_change=N_ITER_NO_CHANGE, tol=TOL,
                random_state=RANDOM_SEED, solver="adam",
            ))
        ])
        pipe.fit(tr[feats].values, tr[TARGET].values)
        y_pred = pipe.predict(te[feats].values)
        for i, (_, row) in enumerate(te.iterrows()):
            loo_rows.append(dict(rock=row["rock"], pct=row["pct"],
                                 h_true=row["h_cm"], h_pred=y_pred[i]))
    return pd.DataFrame(loo_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run & report
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("RAW PAIRS vs MEAN/DIFF  --  LORO Cross-Validation")
print("=" * 60)
print(f"  Both encodings use exactly {len(FEAT_MEANDIFF)} inputs\n")

all_results = {}
for name, feats, diff_cols, swap_pairs in VARIANTS:
    print(f"Running: {name} ...")
    lr = run_loro(df, feats, diff_cols, swap_pairs)
    yt = lr["h_true"].values
    yp = lr["h_pred"].values
    mape = np.mean(np.abs(yt - yp) / np.abs(yt)) * 100
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    print(f"  MAPE = {mape:.1f}%   RMSE = {rmse:.3f} cm")
    all_results[name] = (mape, rmse, yt, yp)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Encoding':<30}  {'MAPE':>7}  {'RMSE':>9}")
print(f"  {'-'*30}  {'-'*7}  {'-'*9}")
for name, (mape, rmse, _, __) in all_results.items():
    print(f"  {name:<30}  {mape:>6.1f}%  {rmse:>7.3f} cm")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle("Raw Pairs (fn_b, fn_s) vs Mean/Diff Encoding\n"
             "LORO CV  |  11 inputs each  |  same MLP (8,4) tanh  |  alpha=1.0",
             fontsize=11, fontweight="bold")

for ax, (name, (mape, rmse, yt, yp)) in zip(axes, all_results.items()):
    lim = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.4, s=45)
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="perfect")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True h (cm)", fontsize=10)
    ax.set_ylabel("Predicted h (cm)", fontsize=10)
    ax.set_title(f"{name}\nMAPE={mape:.1f}%   RMSE={rmse:.3f} cm", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
out_png = OUTDIR / "nn_raw_pairs_comparison.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"\n  Plot saved -> {out_png.name}")
