"""
nn_deepsets.py
==============
Compares three neural network architectures for burial depth prediction:

  1. Baseline sklearn MLP  — current best (11-input mean/diff encoding)
  2. DeepSets (PyTorch)    — shared phi-MLP per direction, mean-pool, then rho-MLP
  3. Siamese (PyTorch)     — same as DeepSets but with difference pooling (concat)

DeepSets reference: Zaheer et al., NeurIPS 2017.
The key advantage over mean/diff: phi can learn nonlinear within-direction
interactions (e.g. fn × Hm × zeta) before pooling, which the linear
mean/diff encoding cannot represent.

Per-direction features fed into phi (5):
    fn, Mp_raw (Hm), damping_ratio (zeta), beta_raw (Sl), mag_gradient (Hg)

Scalar features concatenated after pooling (3):
    w_mean, w_diff, Hs

All models are evaluated with strict Leave-One-Rock-Out (LORO) CV.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "nn_results"
CSV    = BASE / "training_dataset.csv"

# ═════════════════════════════════════════════════════════════════════════════
# TUNABLE SETTINGS  ← edit here
# ═════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Baseline sklearn MLP settings (grid-search best)
HIDDEN_LAYERS    = (8, 4, 2)
ACTIVATION       = "tanh"
ALPHA            = 5.0
MAX_ITER         = 50_000
LEARNING_RATE_SK = 5e-4
N_ITER_NO_CHANGE = 300
TOL              = 1e-6

# DeepSets / Siamese PyTorch settings
PHI_HIDDEN   = 8     # hidden units in shared phi sub-network
RHO_HIDDEN   = (8, 4)  # hidden layer sizes for final rho network
LR           = 1e-3
EPOCHS       = 5_000
PATIENCE     = 300   # early stopping patience (epochs)
WEIGHT_DECAY = 1e-3  # L2 regularisation

TARGET = "h_cm"
# ═════════════════════════════════════════════════════════════════════════════

OUTDIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load raw data  (same pipeline as nn_invariant.py)
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
raw["orient"] = raw["date"].str.split("_").str[-1]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build paired dataset
# ─────────────────────────────────────────────────────────────────────────────
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

    # Per-direction raw features (5 each)
    fn_b  = b["fn_avg_Hz"].mean()
    fn_s  = s["fn_avg_Hz"].mean()
    Mp_b  = b["peak_mag"].mean()
    Mp_s  = s["peak_mag"].mean()
    zb    = b["damping_ratio"].mean()
    zs    = s["damping_ratio"].mean()
    beta_b  = b["log_spatial_slope"].mean()
    beta_s  = s["log_spatial_slope"].mean()
    Hg_b  = b["mag_gradient"].mean()
    Hg_s  = s["mag_gradient"].mean()

    # Mean/diff encoding (for baseline 11-input model)
    w_mean   = (ax1  + ax2)  / 2
    w_diff   = ax2  - ax1
    fn_mean   = (fn_b + fn_s) / 2;  fn_diff   = fn_b - fn_s
    Mp_mean   = (Mp_b + Mp_s) / 2;  Mp_diff   = Mp_b - Mp_s
    zeta_mean = (zb   + zs)   / 2;  zeta_diff = zb   - zs
    beta_mean   = (beta_b + beta_s) / 2;  beta_diff   = beta_b - beta_s

    rows.append(dict(
        rock=rock, pct=pct, date=orient,
        # Raw per-direction features (named <feat>_b / <feat>_s)
        fn_b=fn_b, fn_s=fn_s,
        Mp_b=Mp_b, Mp_s=Mp_s,
        zeta_b=zb, zeta_s=zs,
        beta_b=beta_b, beta_s=beta_s,
        Hg_b=Hg_b, Hg_s=Hg_s,
        # Scalar context
        w_mean=w_mean, w_diff=w_diff, he=Hs,
        # Mean/diff features for baseline
        fn_mean=fn_mean, fn_diff=fn_diff,
        Mp_mean=Mp_mean, Mp_diff=Mp_diff,
        zeta_mean=zeta_mean, zeta_diff=zeta_diff,
        beta_mean=beta_mean, beta_diff=beta_diff,
        h_cm=h,
    ))

df = pd.DataFrame(rows)
print(f"Pairs: {len(df)}   Rocks: {df['rock'].nunique()}\n")

# Baseline 11-input feature set (current best from nn_invariant.py)
FEAT_BASELINE = ["w_mean", "fn_mean", "Mp_mean", "he",
                 "w_diff", "fn_diff", "Mp_diff",
                 "zeta_mean", "zeta_diff", "beta_mean", "beta_diff"]

# Per-direction feature order (used by DeepSets / Siamese)
# Names match the _b / _s column suffixes in the paired dataframe
DIR_FEATS  = ["fn", "Mp", "zeta", "beta", "Hg"]   # 5 features
N_DIR      = len(DIR_FEATS)                         # 5
N_SCALAR   = 3                                      # w_mean, w_diff, he

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch model definitions
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim, hidden_sizes, out_dim, activation=nn.Tanh):
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DeepSetsModel(nn.Module):
    """
    phi: shared MLP applied to each direction independently
    rho: final MLP applied to [pool(phi_b, phi_s), scalars]
    pooling: mean of phi(b) and phi(s)
    """
    def __init__(self, n_dir=5, phi_hidden=8, n_scalar=3, rho_hidden=(8, 4)):
        super().__init__()
        self.phi = make_mlp(n_dir, [phi_hidden], phi_hidden)
        rho_in   = phi_hidden + n_scalar
        self.rho = make_mlp(rho_in, list(rho_hidden), 1)

    def forward(self, x_b, x_s, x_scalar):
        pooled = (self.phi(x_b) + self.phi(x_s)) / 2
        return self.rho(torch.cat([pooled, x_scalar], dim=1)).squeeze(1)


class SiameseModel(nn.Module):
    """
    Same shared phi, but pool by concatenating [phi(b), phi(s)] — preserves
    order information (b vs s). The network can then learn direction asymmetry
    nonlinearly, rather than just via the linear difference.
    """
    def __init__(self, n_dir=5, phi_hidden=8, n_scalar=3, rho_hidden=(8, 4)):
        super().__init__()
        self.phi = make_mlp(n_dir, [phi_hidden], phi_hidden)
        rho_in   = phi_hidden * 2 + n_scalar
        self.rho = make_mlp(rho_in, list(rho_hidden), 1)

    def forward(self, x_b, x_s, x_scalar):
        combined = torch.cat([self.phi(x_b), self.phi(x_s), x_scalar], dim=1)
        return self.rho(combined).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LORO helpers
# ─────────────────────────────────────────────────────────────────────────────

# Diff columns that get sign-flipped for the b/s swap augmentation
DIFF_COLS_ALL = ["w_diff", "fn_diff", "Mp_diff", "Hg_diff", "beta_diff", "zeta_diff"]
# Raw per-direction column pairs that get physically swapped
RAW_SWAP_PAIRS = [("fn_b","fn_s"), ("Mp_b","Mp_s"),
                  ("zeta_b","zeta_s"), ("beta_b","beta_s"), ("Hg_b","Hg_s")]

def augment(df_in):
    """Double dataset by swapping b/s directions.
    - For mean/diff columns: negate all diff columns.
    - For raw per-direction columns: physically swap _b and _s values.
    """
    swap = df_in.copy()
    # Negate diff columns (mean/diff encoding)
    for col in DIFF_COLS_ALL:
        if col in swap.columns:
            swap[col] = -df_in[col]
    # Physically swap raw per-direction columns
    for col_b, col_s in RAW_SWAP_PAIRS:
        if col_b in swap.columns and col_s in swap.columns:
            swap[col_b] = df_in[col_s]
            swap[col_s] = df_in[col_b]
    return pd.concat([df_in, swap], ignore_index=True)


def sklearn_loro(df, feats, target=TARGET):
    """Leave-One-Rock-Out with sklearn Pipeline + b/s swap augmentation."""
    rocks = df["rock"].unique()
    y_true, y_pred = [], []
    for rock in rocks:
        tr = augment(df[df["rock"] != rock])
        te = df[df["rock"] == rock]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION,
                alpha=ALPHA, max_iter=MAX_ITER,
                learning_rate_init=LEARNING_RATE_SK,
                n_iter_no_change=N_ITER_NO_CHANGE, tol=TOL,
                random_state=RANDOM_SEED, solver="adam", early_stopping=False,
            ))
        ])
        pipe.fit(tr[feats].values, tr[target].values)
        y_true.extend(te[target].tolist())
        y_pred.extend(pipe.predict(te[feats].values).tolist())
    return np.array(y_true), np.array(y_pred)


def torch_loro(df, ModelClass, target=TARGET):
    """Leave-One-Rock-Out with a PyTorch DeepSets/Siamese model."""
    rocks  = df["rock"].unique()
    y_true, y_pred = [], []

    dir_cols_b = [f"{f}_b" for f in DIR_FEATS]
    dir_cols_s = [f"{f}_s" for f in DIR_FEATS]
    scalar_cols = ["w_mean", "w_diff", "he"]

    for rock in rocks:
        tr_df = df[df["rock"] != rock].reset_index(drop=True)
        te_df = df[df["rock"] == rock].reset_index(drop=True)

        # Fit scalers on training data
        sc_b      = StandardScaler().fit(tr_df[dir_cols_b].values)
        sc_s      = StandardScaler().fit(tr_df[dir_cols_s].values)
        sc_scalar = StandardScaler().fit(tr_df[scalar_cols].values)
        sc_y      = StandardScaler().fit(tr_df[[target]].values)

        def to_tensors(split_df):
            xb  = torch.tensor(sc_b.transform(split_df[dir_cols_b].values),   dtype=torch.float32)
            xs  = torch.tensor(sc_s.transform(split_df[dir_cols_s].values),   dtype=torch.float32)
            xsc = torch.tensor(sc_scalar.transform(split_df[scalar_cols].values), dtype=torch.float32)
            y   = torch.tensor(sc_y.transform(split_df[[target]].values).ravel(), dtype=torch.float32)
            return xb, xs, xsc, y

        # Augment training: also pass (s, b) swapped — same physical pair, b/s reversed
        tr_aug = augment(tr_df)

        xb_tr, xs_tr, xsc_tr, y_tr = to_tensors(tr_aug)
        xb_te, xs_te, xsc_te, y_te = to_tensors(te_df)

        model = ModelClass(n_dir=N_DIR, phi_hidden=PHI_HIDDEN,
                           n_scalar=N_SCALAR, rho_hidden=RHO_HIDDEN)
        torch.manual_seed(RANDOM_SEED)
        model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

        opt   = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = ReduceLROnPlateau(opt, patience=PATIENCE//4, factor=0.5)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(xb_tr, xs_tr, xsc_tr)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            opt.step()
            sched.step(loss)
            l = loss.item()
            if l < best_loss - 1e-6:
                best_loss = l
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break

        model.eval()
        with torch.no_grad():
            p_scaled = model(xb_te, xs_te, xsc_te).numpy()
        p = sc_y.inverse_transform(p_scaled.reshape(-1, 1)).ravel()

        y_true.extend(te_df[target].tolist())
        y_pred.extend(p.tolist())

    return np.array(y_true), np.array(y_pred)


def metrics(y_true, y_pred, label):
    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"  {label:<28}  MAPE={mape:5.1f}%   RMSE={rmse:.3f} cm")
    return mape, rmse, y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run LORO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ARCHITECTURE COMPARISON  --  LORO Cross-Validation")
print("=" * 60)
print(f"  Per-direction features (phi input): {DIR_FEATS}")
print(f"  Scalar features:                    w_mean, w_diff, Hs")
print(f"  phi hidden: {PHI_HIDDEN}   rho hidden: {RHO_HIDDEN}")
print()

results = {}

print("Running Baseline sklearn MLP (11-input mean/diff) ...")
yt, yp = sklearn_loro(df, FEAT_BASELINE)
results["Baseline\n(11-input sklearn MLP)"] = metrics(yt, yp, "Baseline sklearn MLP (11)")

print("Running DeepSets (PyTorch, mean-pool) ...")
yt, yp = torch_loro(df, DeepSetsModel)
results["DeepSets\n(mean-pool + shared phi)"] = metrics(yt, yp, "DeepSets (mean-pool)")

print("Running Siamese (PyTorch, concat-pool) ...")
yt, yp = torch_loro(df, SiameseModel)
results["Siamese\n(concat-pool + shared phi)"] = metrics(yt, yp, "Siamese (concat-pool)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Architecture Comparison: Mean/Diff Encoding vs DeepSets vs Siamese\n"
             "LORO Cross-Validation  |  phi(fn, Hm, zeta, Sl, Hg) per direction",
             fontsize=12, fontweight="bold")

for ax, (label, (mape, rmse, yt, yp)) in zip(axes, results.items()):
    lims = [0, max(yt.max(), yp.max()) * 1.05]
    ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
    ax.plot(lims, lims, "r--", lw=1.2, label="perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True h (cm)", fontsize=10)
    ax.set_ylabel("Predicted h (cm)", fontsize=10)
    ax.set_title(f"{label}\nMAPE={mape:.1f}%   RMSE={rmse:.3f}cm", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
out_png = OUTDIR / "nn_deepsets_comparison.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"\n  Plot saved -> {out_png.name}")
