#!/usr/bin/env python3
"""
error_characteristics.py
========================
Operational error analysis of the nested-protocol predictions (paper
Section "Error Characteristics" and the error-vs-depth figure).

Reads output/nested_preds_allseeds_meandiff11.csv (produced by nested_cv.py)
and reports: signed bias, MAE, RMSE, 95th-percentile and worst-case error,
failure rates, and a per-burial-depth-quartile breakdown; saves the
error-vs-depth figure.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # repo root
PREDS = BASE / "output" / "nested_preds_allseeds_meandiff11.csv"


def main():
    if not PREDS.exists():
        raise SystemExit(f"{PREDS} not found — run nested_cv.py first.")
    d = pd.read_csv(PREDS)
    d["err"] = d.h_pred - d.h_true
    d["abs_err"] = d.err.abs()

    # Each metric is computed per seed (102 samples) and aggregated as
    # mean +/- std over the 20 seeds, so that every value describes one
    # complete seeded nested-cross-validation evaluation rather than
    # pooling correlated re-predictions.
    def per_seed(fn):
        vals = np.array([fn(g) for _, g in d.groupby("seed")])
        return vals.mean(), vals.std()

    print("=" * 64)
    print("ERROR CHARACTERISTICS — nested LORO, per-seed mean +/- std (0-19)")
    print("=" * 64)
    for label, fn, fmt in [
        ("MAPE", lambda g: (g.abs_err / g.h_true).mean() * 100, "{:.1f} +/- {:.1f} %"),
        ("RMSE", lambda g: np.sqrt((g.err ** 2).mean()), "{:.3f} +/- {:.3f} cm"),
        ("MAE", lambda g: g.abs_err.mean(), "{:.3f} +/- {:.3f} cm"),
        ("signed bias", lambda g: g.err.mean(), "{:+.3f} +/- {:.3f} cm"),
        ("median error", lambda g: g.err.median(), "{:+.3f} +/- {:.3f} cm"),
        ("P95 |error|", lambda g: np.percentile(g.abs_err, 95), "{:.2f} +/- {:.2f} cm"),
        ("worst |error| per model", lambda g: g.abs_err.max(), "{:.2f} +/- {:.2f} cm"),
        ("fail |rel|>25%", lambda g: (g.abs_err / g.h_true > 0.25).mean() * 100, "{:.1f} +/- {:.1f} %"),
        ("fail |abs|>2cm", lambda g: (g.abs_err > 2).mean() * 100, "{:.1f} +/- {:.1f} %"),
    ]:
        m, s = per_seed(fn)
        print(f"  {label:24s}: {fmt.format(m, s)}")
    print(f"  {'largest |error| any run':24s}: {d.abs_err.max():.2f} cm")

    qs = pd.qcut(d.h_true, 4)
    d["q"] = qs
    print("\n  By burial-depth quartile (per-seed mean +/- std):")
    print(f"    {'range (cm)':>16} {'MAPE%':>13} {'bias (cm)':>15} {'RMSE (cm)':>13}")
    for iv in d.q.cat.categories:
        sub = d[d.q == iv]
        per = []
        for _, g in sub.groupby("seed"):
            per.append(((g.abs_err / g.h_true).mean() * 100,
                        g.err.mean(), np.sqrt((g.err ** 2).mean())))
        per = np.array(per)
        print(f"    {str(iv):>16} "
              f"{per[:,0].mean():6.1f}+/-{per[:,0].std():4.1f} "
              f"{per[:,1].mean():+7.2f}+/-{per[:,1].std():4.2f} "
              f"{per[:,2].mean():6.2f}+/-{per[:,2].std():4.2f}")

    # figure (paper Fig. 5): single seeded nested-CV evaluation (seed 0) — scatter,
    # quartile mean +/- std, Bland-Altman-style bias and 95% LoA lines
    s0 = d[d.seed == 0]
    bias = s0.err.mean()
    sd = s0.err.std()
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    print(f"\n  seed-0 95% LoA: {loa_lo:+.2f} to {loa_hi:+.2f} cm")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.axhline(bias, color="#cc3333", lw=1.1, ls="--", zorder=2,
               label=f"bias = {bias:+.2f} cm")
    ax.axhline(loa_hi, color="#444", lw=0.9, ls=":", zorder=2,
               label=f"$\\pm$1.96 SD = {loa_lo:+.2f}, {loa_hi:+.2f} cm")
    ax.axhline(loa_lo, color="#444", lw=0.9, ls=":", zorder=2)
    ax.scatter(s0.h_true, s0.err, s=26, alpha=0.6, color="#3a7ebf",
               edgecolors="#1c4f8a", linewidths=0.5, zorder=3,
               label="seed-0 predictions")
    grp0 = s0.groupby(pd.qcut(s0.h_true, 4), observed=True)
    centers = np.array([g.h_true.mean() for _, g in grp0])
    means = np.array([g.err.mean() for _, g in grp0])
    stds = np.array([g.err.std() for _, g in grp0])
    ax.errorbar(centers, means, yerr=stds, fmt="s-", color="#cc3333", lw=1.2,
                ms=5, capsize=3, zorder=4, label="quartile mean $\\pm$ std")
    ax.set_xlabel("True burial depth $h$ (cm)", fontsize=7)
    ax.set_ylabel("Prediction error $\\hat{h}-h$ (cm)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9)
    out = BASE / "output" / "error_vs_depth.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
