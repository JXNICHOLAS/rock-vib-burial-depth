#!/usr/bin/env python3
"""
plot_nested_results.py
======================
Regenerates the paper's nested-protocol figures from nested_cv.py output:

  output/paper_results_h_nested.png  — pred-vs-true burial depth (Fig. 4a)
  output/paper_results_V_nested.png  — pred-vs-true total volume (Fig. 4b)
  output/error_vs_depth.png          — signed error vs depth with bias and
                                       95% limits-of-agreement lines (Fig. 5)

Requires output/nested_preds_allseeds_meandiff.csv (run nested_cv.py first).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from paired_dataset import load_paired

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
PREDS = OUT / "nested_preds_allseeds_meandiff.csv"


def mape(t, p):
    t = np.asarray(t); p = np.asarray(p)
    return float(np.mean(np.abs(p - t) / t) * 100)


def rmse(t, p):
    t = np.asarray(t); p = np.asarray(p)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def make_panel(ax, x_true, y_pred, xlabel, ylabel):
    x_true = np.asarray(x_true); y_pred = np.asarray(y_pred)
    data_max = max(x_true.max(), y_pred.max())
    raw_step = data_max / 5.0
    mag = 10 ** np.floor(np.log10(raw_step))
    tick_step = float(mag * min([1, 2, 5, 10], key=lambda f: abs(f * mag - raw_step)))
    lim_hi = float(np.ceil(data_max / (tick_step / 2)) * (tick_step / 2))
    xs = np.array([0.0, lim_hi])
    ax.plot(xs, xs * 1.25, ':', color='#cc3333', linewidth=0.9, label=r'$\pm$25%')
    ax.plot(xs, xs * 0.75, ':', color='#cc3333', linewidth=0.9)
    ax.plot(xs, xs, 'r--', linewidth=1.2, label='Perfect')
    ax.scatter(x_true, y_pred, s=30, alpha=0.65, edgecolors='#1c4f8a',
               linewidths=0.5, color='#3a7ebf', zorder=3)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlim(0.0, lim_hi)
    ax.set_ylim(0.0, lim_hi)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, loc='upper left')


def main():
    if not PREDS.exists():
        raise SystemExit(f"{PREDS} not found — run nested_cv.py first.")
    preds = pd.read_csv(PREDS)
    dims = load_paired()[["rock", "pct", "date", "axis1_cm", "axis2_cm", "he_cm"]]
    preds = preds.merge(dims, on=["rock", "pct", "date"], how="left")

    preds["V_true"] = preds.axis1_cm * preds.axis2_cm * (preds.he_cm + preds.h_true)
    preds["V_pred"] = preds.axis1_cm * preds.axis2_cm * (preds.he_cm + preds.h_pred)
    s0 = preds[preds.seed == 0].copy()

    print(f"seed0 h: MAPE={mape(s0.h_true, s0.h_pred):.1f}%  RMSE={rmse(s0.h_true, s0.h_pred):.2f} cm")
    print(f"seed0 V: MAPE={mape(s0.V_true, s0.V_pred):.1f}%  RMSE={rmse(s0.V_true, s0.V_pred):.0f} cm3")

    for x, y, xl, yl, fname in [
        (s0.h_true, s0.h_pred, 'True $h$ (cm)', 'Predicted $h$ (cm)',
         'paper_results_h_nested.png'),
        (s0.V_true, s0.V_pred, r'True $V$ (cm$^3$)', r'Predicted $V$ (cm$^3$)',
         'paper_results_V_nested.png'),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
        make_panel(ax, x, y, xl, yl)
        fig.savefig(OUT / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved -> {OUT / fname}")

    # error vs depth for a single trained model (seed 0), with
    # Bland-Altman-style bias + limits of agreement (matches paper Fig. 5)
    s0["err"] = s0.h_pred - s0.h_true
    grp = s0.groupby(pd.qcut(s0.h_true, 4), observed=True)
    bias = s0["err"].mean()
    sd = s0["err"].std()
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

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
    centers = np.array([g.h_true.mean() for _, g in grp])
    means = np.array([g.err.mean() for _, g in grp])
    stds = np.array([g.err.std() for _, g in grp])
    ax.errorbar(centers, means, yerr=stds, fmt="s-", color="#cc3333", lw=1.2,
                ms=5, capsize=3, zorder=4, label="quartile mean $\\pm$ std")
    ax.set_xlabel("True burial depth $h$ (cm)", fontsize=7)
    ax.set_ylabel("Prediction error $\\hat{h}-h$ (cm)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9)
    fig.savefig(OUT / "error_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT / 'error_vs_depth.png'}")


if __name__ == "__main__":
    main()
