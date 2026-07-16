"""
nn_transformer.py
=================
Tests a Transformer architecture on raw per-direction inputs vs the
current best mean/diff baseline.

Architecture
------------
Each direction (b, s) becomes one token of dimension d_model:
    token = Linear(n_dir_features -> d_model) + learned direction embedding

A TransformerEncoder applies multi-head self-attention over the 2 tokens,
letting each direction "see" the other before forming its representation.

    token_b_ctx, token_s_ctx = TransformerEncoder([token_b, token_s])

The contextualised tokens are mean-pooled, then concatenated with scalar
context features (ax1, ax2, Hs) and passed through a final regression MLP.

Key difference vs Siamese: in Siamese, phi(b) and phi(s) are processed
independently. In a Transformer, attention lets each token attend to the
other — cross-direction interactions happen inside the representation step.

Models compared
---------------
  1. Baseline sklearn MLP   — 11-input mean/diff (current best, 24.9% MAPE)
  2. Transformer-S           — d_model=16, 1 layer, 2 heads (small)
  3. Transformer-M           — d_model=32, 2 layers, 4 heads (medium)
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
# TUNABLE SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Baseline sklearn MLP
HIDDEN_LAYERS    = (8, 4, 2)
ACTIVATION       = "tanh"
ALPHA            = 5.0
MAX_ITER         = 50_000
LEARNING_RATE_SK = 5e-4
N_ITER_NO_CHANGE = 300
TOL              = 1e-6

# Transformer training
LR           = 1e-3
EPOCHS       = 12_000
PATIENCE     = 500
WEIGHT_DECAY = 2e-3
DROPOUT      = 0.1

TARGET = "h_cm"
# ═════════════════════════════════════════════════════════════════════════════

OUTDIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
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

    fn_b = b["fn_avg_Hz"].mean();     fn_s = s["fn_avg_Hz"].mean()
    Mp_b = b["peak_mag"].mean();      Mp_s = s["peak_mag"].mean()
    zb   = b["damping_ratio"].mean();  zs  = s["damping_ratio"].mean()
    beta_b = b["log_spatial_slope"].mean(); beta_s = s["log_spatial_slope"].mean()
    Hg_b = b["mag_gradient"].mean();  Hg_s = s["mag_gradient"].mean()

    rows.append(dict(
        rock=rock, pct=pct, date=orient,
        ax1=ax1, ax2=ax2, he=Hs,
        # Mean/diff
        w_mean=(ax1+ax2)/2,    w_diff=ax2-ax1,
        fn_mean=(fn_b+fn_s)/2,  fn_diff=fn_b-fn_s,
        Mp_mean=(Mp_b+Mp_s)/2, Mp_diff=Mp_b-Mp_s,
        zeta_mean=(zb+zs)/2,    zeta_diff=zb-zs,
        beta_mean=(beta_b+beta_s)/2,  beta_diff=beta_b-beta_s,
        # Raw per-direction (5 features each)
        fn_b=fn_b, fn_s=fn_s,
        Mp_b=Mp_b, Mp_s=Mp_s,
        zb=zb, zs=zs,
        beta_b=beta_b, beta_s=beta_s,
        Hg_b=Hg_b, Hg_s=Hg_s,
        h_cm=h,
    ))

df = pd.DataFrame(rows)
print(f"Pairs: {len(df)}   Rocks: {df['rock'].nunique()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────
DIFF_COLS  = ["w_diff","fn_diff","Mp_diff","zeta_diff","beta_diff"]
SWAP_PAIRS = [("fn_b","fn_s"),("Mp_b","Mp_s"),("zb","zs"),
              ("beta_b","beta_s"),("Hg_b","Hg_s"),("ax1","ax2")]

def augment(df_in):
    swap = df_in.copy()
    for col in DIFF_COLS:
        if col in swap.columns:
            swap[col] = -df_in[col]
    for ca, cb in SWAP_PAIRS:
        if ca in swap.columns:
            swap[ca] = df_in[cb]
            swap[cb] = df_in[ca]
    return pd.concat([df_in, swap], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model definitions
# ─────────────────────────────────────────────────────────────────────────────
N_DIR_FEATS = 4   # fd, Hm, z, Sl  per direction  (Hg excluded to match baseline)
N_SCALAR    = 3   # ax1, ax2, Hs

DIR_COLS_B = ["fn_b", "Mp_b", "zb",  "beta_b"]
DIR_COLS_S = ["fn_s", "Mp_s", "zs",  "beta_s"]
SCALAR_COLS = ["ax1", "ax2", "he"]
FEAT_MEANDIFF = ["w_mean","fn_mean","Mp_mean","he",
                 "w_diff","fn_diff","Mp_diff",
                 "zeta_mean","zeta_diff","beta_mean","beta_diff"]


class TransformerRegressor(nn.Module):
    """
    Two tokens (b-direction, s-direction) processed by a Transformer encoder.
    Each token = embed(raw features) + learned direction type embedding.
    Tokens are mean-pooled after attention, then concatenated with scalars.
    """
    def __init__(self, n_dir=5, d_model=16, n_heads=2, n_layers=1,
                 ffn_dim=32, n_scalar=3, mlp_hidden=(16, 8), dropout=0.1):
        super().__init__()
        self.embed    = nn.Linear(n_dir, d_model)
        # Learned type embeddings: index 0 = b-direction, index 1 = s-direction
        self.type_emb = nn.Embedding(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Final regression head: pooled token + scalars
        rho_in = d_model + n_scalar
        layers = []
        prev = rho_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_b, x_s, x_scalar):
        # Embed each direction: (batch, d_model)
        tok_b = self.embed(x_b) + self.type_emb(torch.zeros(x_b.size(0), dtype=torch.long, device=x_b.device))
        tok_s = self.embed(x_s) + self.type_emb(torch.ones( x_s.size(0), dtype=torch.long, device=x_s.device))
        # Stack into sequence of length 2: (batch, 2, d_model)
        seq = torch.stack([tok_b, tok_s], dim=1)
        # Transformer: each token attends to both tokens
        out = self.transformer(seq)            # (batch, 2, d_model)
        # Mean-pool over the 2 tokens
        pooled = out.mean(dim=1)               # (batch, d_model)
        # Concat scalars and regress
        return self.head(torch.cat([pooled, x_scalar], dim=1)).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LORO runners
# ─────────────────────────────────────────────────────────────────────────────
def sklearn_loro(df, feats):
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
                random_state=RANDOM_SEED, solver="adam",
            ))
        ])
        pipe.fit(tr[feats].values, tr[TARGET].values)
        y_true.extend(te[TARGET].tolist())
        y_pred.extend(pipe.predict(te[feats].values).tolist())
    return np.array(y_true), np.array(y_pred)


def transformer_loro(df, d_model, n_heads, n_layers, ffn_dim, mlp_hidden):
    rocks = df["rock"].unique()
    y_true, y_pred = [], []

    for rock in rocks:
        tr_df = augment(df[df["rock"] != rock]).reset_index(drop=True)
        te_df = df[df["rock"] == rock].reset_index(drop=True)

        sc_b      = StandardScaler().fit(tr_df[DIR_COLS_B].values)
        sc_s      = StandardScaler().fit(tr_df[DIR_COLS_S].values)
        sc_scalar = StandardScaler().fit(tr_df[SCALAR_COLS].values)
        sc_y      = StandardScaler().fit(tr_df[[TARGET]].values)

        def to_t(split_df):
            xb  = torch.tensor(sc_b.transform(split_df[DIR_COLS_B].values),      dtype=torch.float32)
            xs  = torch.tensor(sc_s.transform(split_df[DIR_COLS_S].values),      dtype=torch.float32)
            xsc = torch.tensor(sc_scalar.transform(split_df[SCALAR_COLS].values),dtype=torch.float32)
            y   = torch.tensor(sc_y.transform(split_df[[TARGET]].values).ravel(), dtype=torch.float32)
            return xb, xs, xsc, y

        xb_tr, xs_tr, xsc_tr, y_tr = to_t(tr_df)
        xb_te, xs_te, xsc_te, _    = to_t(te_df)

        torch.manual_seed(RANDOM_SEED)
        model = TransformerRegressor(
            n_dir=N_DIR_FEATS, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, ffn_dim=ffn_dim,
            n_scalar=N_SCALAR, mlp_hidden=mlp_hidden, dropout=DROPOUT,
        )
        opt   = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = ReduceLROnPlateau(opt, patience=PATIENCE//4, factor=0.5)
        loss_fn = nn.MSELoss()

        best_loss, no_imp = float("inf"), 0
        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xb_tr, xs_tr, xsc_tr), y_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(loss.detach())
            l = loss.item()
            if l < best_loss - 1e-6:
                best_loss, no_imp = l, 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE:
                break

        model.eval()
        with torch.no_grad():
            p_sc = model(xb_te, xs_te, xsc_te).numpy()
        p = sc_y.inverse_transform(p_sc.reshape(-1, 1)).ravel()
        y_true.extend(te_df[TARGET].tolist())
        y_pred.extend(p.tolist())

    return np.array(y_true), np.array(y_pred)


def report(label, yt, yp):
    mape = np.mean(np.abs(yt - yp) / np.abs(yt)) * 100
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    print(f"  {label:<38}  MAPE={mape:5.1f}%   RMSE={rmse:.3f} cm")
    return mape, rmse


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Run
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TRANSFORMER vs BASELINE  --  LORO Cross-Validation")
print("=" * 70)
print(f"  Transformer input: raw per-direction (4 each): {DIR_COLS_B} + scalars ax1,ax2,Hs")
print(f"  Baseline input:    11-input mean/diff encoding")
print()

results = {}

print("Running Baseline sklearn MLP (11-input mean/diff) ...")
yt, yp = sklearn_loro(df, FEAT_MEANDIFF)
results["Baseline\n(11-input mean/diff,\nsklearn MLP 8-4)"] = (*report("Baseline sklearn MLP (11)", yt, yp), yt, yp)

print("Running Transformer-S  (d=16, 1 layer, 2 heads) ...")
yt, yp = transformer_loro(df, d_model=16, n_heads=2, n_layers=1,
                           ffn_dim=32, mlp_hidden=(16, 8))
results["Transformer-S\n(d=16, 1 layer, 2 heads,\nraw inputs)"] = (*report("Transformer-S (d=16, 1L, 2H)", yt, yp), yt, yp)

print("Running Transformer-M  (d=32, 2 layers, 4 heads) ...")
yt, yp = transformer_loro(df, d_model=32, n_heads=4, n_layers=2,
                           ffn_dim=64, mlp_hidden=(32, 16))
results["Transformer-M\n(d=32, 2 layers, 4 heads,\nraw inputs)"] = (*report("Transformer-M (d=32, 2L, 4H)", yt, yp), yt, yp)

print("Running Transformer-L  (d=64, 4 layers, 4 heads) ...")
yt, yp = transformer_loro(df, d_model=64, n_heads=4, n_layers=4,
                           ffn_dim=128, mlp_hidden=(64, 32, 16))
results["Transformer-L\n(d=64, 4 layers, 4 heads,\nraw inputs)"] = (*report("Transformer-L (d=64, 4L, 4H)", yt, yp), yt, yp)

print("Running Transformer-XL  (d=128, 6 layers, 8 heads) ...")
yt, yp = transformer_loro(df, d_model=128, n_heads=8, n_layers=6,
                           ffn_dim=256, mlp_hidden=(128, 64, 32))
results["Transformer-XL\n(d=128, 6 layers, 8 heads,\nraw inputs)"] = (*report("Transformer-XL (d=128, 6L, 8H)", yt, yp), yt, yp)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Transformer Scaling: S / M / L / XL vs Mean/Diff Baseline\n"
             "LORO CV  |  augmentation: b/s swap  |  weight_decay=1e-3",
             fontsize=11, fontweight="bold")
axes = axes.flatten()

for ax, (label, (mape, rmse, yt, yp)) in zip(axes, results.items()):
    lim = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="perfect")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True h (cm)", fontsize=10)
    ax.set_ylabel("Predicted h (cm)", fontsize=10)
    ax.set_title(f"{label}\nMAPE={mape:.1f}%   RMSE={rmse:.3f}cm", fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
out_png = OUTDIR / "nn_transformer_comparison.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"\n  Plot saved -> {out_png.name}")
