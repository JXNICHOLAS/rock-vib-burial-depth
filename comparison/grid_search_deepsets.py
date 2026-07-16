"""
grid_search_deepsets.py
=======================
Grid search over PHI_HIDDEN, RHO_HIDDEN, and WEIGHT_DECAY for
DeepSets and Siamese models, averaged over N_SEEDS random seeds.

Usage:
  python grid_search_deepsets.py --model deepsets
  python grid_search_deepsets.py --model siamese

Search space (27 configs × N_SEEDS seeds per model):
  phi_hidden  : [4, 8, 16]
  rho_hidden  : [(4,), (8, 4), (16, 8)]
  weight_decay: [1e-3, 5e-3, 1e-2]
  seeds       : [0, 42, 7]
"""

import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CSV  = BASE / "training_dataset.csv"

# ── CLI argument ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["deepsets", "siamese"], required=True)
args = parser.parse_args()

# ── Fixed training settings ──────────────────────────────────────────────────
LR       = 1e-3
EPOCHS   = 5_000
PATIENCE = 300
TARGET   = "h_cm"
N_SEEDS  = 3
SEEDS    = [0, 42, 7]

# ── Grid ─────────────────────────────────────────────────────────────────────
PHI_HIDDENS   = [4, 8, 16]
RHO_HIDDENS   = [(4,), (8, 4), (16, 8)]
WEIGHT_DECAYS = [1e-3, 5e-3, 1e-2]

# ── Feature layout ───────────────────────────────────────────────────────────
DIR_FEATS   = ["fn", "Mp", "zeta", "beta", "Hg"]
N_DIR       = len(DIR_FEATS)        # 5
N_SCALAR    = 3                     # w_mean, w_diff, he

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
raw["orient"] = raw["date"].str.split("_").str[-1]
print(f"  {len(raw)} rows retained.\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build paired dataset
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for (rock, pct, orient), grp in raw.groupby(["rock", "burial_pct", "orient"]):
    b = grp[grp["direction"] == "b"]
    s = grp[grp["direction"] == "s"]
    if len(b) == 0 or len(s) == 0:
        continue
    rows.append(dict(
        rock=rock, pct=pct,
        fn_b=b["fn_avg_Hz"].mean(),       fn_s=s["fn_avg_Hz"].mean(),
        Mp_b=b["peak_mag"].mean(),         Mp_s=s["peak_mag"].mean(),
        zeta_b=b["damping_ratio"].mean(),  zeta_s=s["damping_ratio"].mean(),
        beta_b=b["log_spatial_slope"].mean(), beta_s=s["log_spatial_slope"].mean(),
        Hg_b=b["mag_gradient"].mean(),    Hg_s=s["mag_gradient"].mean(),
        w_mean=(grp["axis1_cm"].iloc[0] + grp["axis2_cm"].iloc[0]) / 2,
        w_diff=grp["axis2_cm"].iloc[0]    - grp["axis1_cm"].iloc[0],
        he=grp["Hs_cm"].iloc[0],
        h_cm=grp["h_cm"].iloc[0],
    ))

df = pd.DataFrame(rows)
print(f"Pairs: {len(df)}   Rocks: {df['rock'].nunique()}\n")

RAW_SWAP_PAIRS = [("fn_b","fn_s"), ("Mp_b","Mp_s"),
                  ("zeta_b","zeta_s"), ("beta_b","beta_s"), ("Hg_b","Hg_s")]

def augment(df_in):
    swap = df_in.copy()
    for ca, cb in RAW_SWAP_PAIRS:
        swap[ca] = df_in[cb]
        swap[cb] = df_in[ca]
    return pd.concat([df_in, swap], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model definitions
# ─────────────────────────────────────────────────────────────────────────────
def make_mlp(in_dim, hidden_sizes, out_dim, activation=nn.Tanh):
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DeepSetsModel(nn.Module):
    def __init__(self, n_dir=5, phi_hidden=8, n_scalar=3, rho_hidden=(8, 4)):
        super().__init__()
        self.phi = make_mlp(n_dir, [phi_hidden], phi_hidden)
        self.rho = make_mlp(phi_hidden + n_scalar, list(rho_hidden), 1)

    def forward(self, x_b, x_s, x_scalar):
        pooled = (self.phi(x_b) + self.phi(x_s)) / 2
        return self.rho(torch.cat([pooled, x_scalar], dim=1)).squeeze(1)


class SiameseModel(nn.Module):
    def __init__(self, n_dir=5, phi_hidden=8, n_scalar=3, rho_hidden=(8, 4)):
        super().__init__()
        self.phi = make_mlp(n_dir, [phi_hidden], phi_hidden)
        self.rho = make_mlp(phi_hidden * 2 + n_scalar, list(rho_hidden), 1)

    def forward(self, x_b, x_s, x_scalar):
        combined = torch.cat([self.phi(x_b), self.phi(x_s), x_scalar], dim=1)
        return self.rho(combined).squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LORO runner for one config / seed
# ─────────────────────────────────────────────────────────────────────────────
dir_cols_b  = [f"{f}_b" for f in DIR_FEATS]
dir_cols_s  = [f"{f}_s" for f in DIR_FEATS]
scalar_cols = ["w_mean", "w_diff", "he"]


def run_loro(ModelClass, phi_hidden, rho_hidden, weight_decay, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rocks = df["rock"].unique()
    y_true, y_pred = [], []

    for rock in rocks:
        tr_df = augment(df[df["rock"] != rock]).reset_index(drop=True)
        te_df = df[df["rock"] == rock].reset_index(drop=True)

        sc_b      = StandardScaler().fit(tr_df[dir_cols_b].values)
        sc_s      = StandardScaler().fit(tr_df[dir_cols_s].values)
        sc_scalar = StandardScaler().fit(tr_df[scalar_cols].values)
        sc_y      = StandardScaler().fit(tr_df[[TARGET]].values)

        def to_tensors(split_df):
            xb  = torch.tensor(sc_b.transform(split_df[dir_cols_b].values),       dtype=torch.float32)
            xs  = torch.tensor(sc_s.transform(split_df[dir_cols_s].values),       dtype=torch.float32)
            xsc = torch.tensor(sc_scalar.transform(split_df[scalar_cols].values), dtype=torch.float32)
            y   = torch.tensor(sc_y.transform(split_df[[TARGET]].values).ravel(), dtype=torch.float32)
            return xb, xs, xsc, y

        xb_tr, xs_tr, xsc_tr, y_tr = to_tensors(tr_df)
        xb_te, xs_te, xsc_te, _    = to_tensors(te_df)

        torch.manual_seed(seed)
        model = ModelClass(n_dir=N_DIR, phi_hidden=phi_hidden,
                           n_scalar=N_SCALAR, rho_hidden=rho_hidden)

        opt   = Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        sched = ReduceLROnPlateau(opt, patience=PATIENCE//4, factor=0.5)
        loss_fn = nn.MSELoss()

        best_loss, no_improve = float("inf"), 0
        for _ in range(EPOCHS):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xb_tr, xs_tr, xsc_tr), y_tr)
            loss.backward()
            opt.step()
            sched.step(loss)
            l = loss.item()
            if l < best_loss - 1e-6:
                best_loss, no_improve = l, 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break

        model.eval()
        with torch.no_grad():
            p_scaled = model(xb_te, xs_te, xsc_te).numpy()
        p = sc_y.inverse_transform(p_scaled.reshape(-1, 1)).ravel()
        y_true.extend(te_df[TARGET].tolist())
        y_pred.extend(p.tolist())

    yt, yp = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs(yt - yp) / np.abs(yt)) * 100
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    return mape, rmse

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Grid search
# ─────────────────────────────────────────────────────────────────────────────
configs = list(itertools.product(PHI_HIDDENS, RHO_HIDDENS, WEIGHT_DECAYS))
total   = len(configs)

model_map = {"deepsets": ("DeepSets", DeepSetsModel), "siamese": ("Siamese", SiameseModel)}
for model_name, ModelClass in [model_map[args.model]]:
    print(f"\n{'='*65}")
    print(f"  GRID SEARCH — {model_name}  ({total} configs x {N_SEEDS} seeds)")
    print(f"{'='*65}")

    best_mape = float("inf")
    best_cfg  = None
    rows_out  = []

    for i, (phi, rho, wd) in enumerate(configs, 1):
        mapes, rmses = [], []
        for seed in SEEDS:
            m, r = run_loro(ModelClass, phi, rho, wd, seed)
            mapes.append(m);  rmses.append(r)

        avg_mape = np.mean(mapes)
        avg_rmse = np.mean(rmses)
        rows_out.append(dict(phi=phi, rho=str(rho), wd=wd,
                             avg_mape=avg_mape, avg_rmse=avg_rmse))

        marker = " <-- best" if avg_mape < best_mape else ""
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_cfg  = (phi, rho, wd)

        print(f"  [{i:2d}/{total}] phi={phi:2d}  rho={str(rho):<10}  wd={wd:.0e}"
              f"  -> MAPE={avg_mape:5.1f}%  RMSE={avg_rmse:.3f}cm{marker}", flush=True)

    print(f"\n  *** Best {model_name}: phi={best_cfg[0]}  rho={best_cfg[1]}"
          f"  wd={best_cfg[2]:.0e}  MAPE={best_mape:.1f}% ***\n")

    out_csv = BASE / "comparison" / f"grid_search_{model_name.lower()}_results.csv"
    pd.DataFrame(rows_out).sort_values("avg_mape").to_csv(out_csv, index=False)
    print(f"  Results saved -> {out_csv.name}")
