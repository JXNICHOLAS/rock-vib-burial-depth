"""
make_training_dataset.py
========================
Scans Training_data/ for all NPZ files, extracts stored metadata and spectrum,
computes derived features, and writes training_dataset.csv.

Run after PlotAverageSpectrumFromSVD_Batch.m has finished generating NPZs.

Computed columns:
    axis1_cm, axis2_cm, axis3_cm  — dimensions from size_x/y/z_mm  [cm]
    h_cm       — buried depth    = burial_pct/100 * vertical_axis_cm
    Hs_cm      — exposed height  = vertical_axis_cm - h_cm
    M_kg       — weight_g / 1000
    density_g_cm3 — weight_g / (ax1*ax2*ax3)  [g/cm^3]
    rock       — 'r' + int(rock_num)
    date       — 'MM_DD_axis'
    direction  — NPZ 'face' field  (b or s)
    mag_gradient — max |d(magnitude)/d(freq)|   sharpest spectral slope
    mag_spatial_ratio — top2_peak_mag / bottom_peak_mag  (NaN for old NPZs)
    mag_top, mag_bottom, n_scan_points — supporting spatial fields
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE         = Path(__file__).resolve().parent
TRAINING_DIR = BASE / "NPZ_data"
CSV_OUT      = BASE / "training_dataset.csv"

npz_files = sorted(TRAINING_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} NPZ files in {TRAINING_DIR}")

rows = []
skipped = 0

for npz_path in npz_files:
    try:
        d = np.load(npz_path, allow_pickle=True)

        # ── Required fields ─────────────────────────────────────────────────
        def _f(key, fallback=np.nan):
            """Safely extract a scalar float from the NPZ."""
            if key not in d:
                return fallback
            val = d[key]
            # stored as 0-d or 1-d array
            val = float(np.asarray(val).flat[0]) if np.asarray(val).size > 0 else fallback
            # jsonencode writes null as the Python string "null"
            if isinstance(val, str):
                return fallback
            return val

        def _s(key, fallback=""):
            """Safely extract a scalar string from the NPZ."""
            if key not in d:
                return fallback
            val = np.asarray(d[key])
            s   = str(val.flat[0]) if val.size > 0 else fallback
            return "" if s in ("", "nan", "None", "null") else s

        rock_num   = _f("rock_num")
        size_x_mm  = _f("size_x_mm")
        size_y_mm  = _f("size_y_mm")
        size_z_mm  = _f("size_z_mm")
        weight_g   = _f("weight_g")
        month      = _f("month")
        day        = _f("day")
        axis       = _s("axis")
        burial_pct = _f("burial_pct")
        face       = _s("face")
        run_num    = _f("run_num")

        fn_peak_Hz       = _f("fn_peak_Hz")
        fn_halfpower_Hz  = _f("fn_halfpower_Hz")
        f1_Hz            = _f("f1_Hz")
        f2_Hz            = _f("f2_Hz")
        damping_ratio    = _f("damping_ratio")
        halfpower_valid  = _f("halfpower_valid")

        freq_hz   = np.asarray(d["frequency_Hz"], dtype=float)
        magnitude = np.asarray(d["magnitude"],    dtype=float)

        # ── Skip if critical fields are missing ──────────────────────────────
        if any(np.isnan(v) for v in [rock_num, size_x_mm, size_y_mm, size_z_mm,
                                      weight_g, burial_pct, fn_peak_Hz]):
            print(f"  SKIP (missing fields): {npz_path.name}")
            skipped += 1
            continue
        if axis not in ("x", "z") or face not in ("b", "s"):
            print(f"  SKIP (bad axis/face '{axis}'/'{face}'): {npz_path.name}")
            skipped += 1
            continue

        # ── Derived geometry ─────────────────────────────────────────────────
        ax1 = size_x_mm / 10.0    # [cm]
        ax2 = size_y_mm / 10.0
        ax3 = size_z_mm / 10.0

        # Vertical axis drives burial depth
        vert_cm = ax3 if axis == "z" else ax1
        h_cm    = (burial_pct / 100.0) * vert_cm
        Hs_cm   = vert_cm - h_cm
        M_kg    = weight_g / 1000.0
        vol_cm3 = ax1 * ax2 * ax3
        density = weight_g / vol_cm3   # g/cm^3

        rock    = f"r{int(round(rock_num))}"
        date    = f"{int(round(month)):02d}_{int(round(day)):02d}_{axis}"

        # ── Spectral gradient (max |dMag/dFreq|) ────────────────────────────
        if len(freq_hz) > 1:
            dM = np.diff(magnitude)
            df = np.diff(freq_hz)
            df[df == 0] = np.nan
            mag_gradient = float(np.nanmax(np.abs(dM / df)))
        else:
            mag_gradient = np.nan

        # ── Spatial gradient (new NPZs only) ────────────────────────────────
        mag_spatial_ratio   = _f("mag_spatial_ratio")
        spatial_slope       = _f("spatial_slope")   # linear fit slope (FRF unit)/mm
        spatial_r2          = _f("spatial_r2")       # linear fit R²
        mag_top             = _f("mag_top")
        mag_bottom          = _f("mag_bottom")
        y_top_mm            = _f("y_top_mm")
        y_bottom_mm         = _f("y_bottom_mm")
        n_scan_points       = _f("n_scan_points")
        # Normalised gradient: (top_mag - bot_mag) / vertical_span_mm
        dy_mm = y_top_mm - y_bottom_mm
        if not np.isnan(dy_mm) and dy_mm > 0 and not np.isnan(mag_top) and not np.isnan(mag_bottom):
            spatial_grad_norm = (mag_top - mag_bottom) / dy_mm
            # Log-slope: e-folds per mm (scale-invariant exponential decay constant)
            # log(top/bot) / dy  — positive = mag decays toward sand as expected
            if mag_bottom > 0 and mag_top > 0:
                log_spatial_slope = np.log(mag_top / mag_bottom) / dy_mm
            else:
                log_spatial_slope = np.nan
        else:
            spatial_grad_norm = np.nan
            log_spatial_slope = np.nan

        # ── Phase features (NPZs regenerated after MATLAB update) ───────────
        phase_at_fn_avg_deg      = _f("phase_at_fn_avg_deg")
        phase_spread_deg         = _f("phase_spread_deg")
        phase_slope_deg_per_mm   = _f("phase_slope_deg_per_mm")
        phase_flip_flag          = _f("phase_flip_flag")
        n_phase_points           = _f("n_phase_points")

        # Per-point phase array → phase difference top-to-bottom
        pt_phases = d["point_phases_deg"] if "point_phases_deg" in d else None
        pt_ymm_ph = d["point_ymm_ph"]     if "point_ymm_ph"     in d else None
        phase_top_bot_diff_deg = np.nan
        if pt_phases is not None and pt_ymm_ph is not None:
            ph_arr = np.asarray(pt_phases, dtype=float)
            y_arr  = np.asarray(pt_ymm_ph, dtype=float)
            valid  = ~np.isnan(ph_arr) & ~np.isnan(y_arr)
            if valid.sum() >= 2:
                ph_sorted = ph_arr[valid][np.argsort(y_arr[valid])]
                # Phase difference: top point minus bottom point (wrapped to [-180,180])
                diff_raw = float(ph_sorted[-1] - ph_sorted[0])
                phase_top_bot_diff_deg = (diff_raw + 180) % 360 - 180

        rows.append(dict(
            filename          = npz_path.name,
            npz_file          = str(npz_path),
            rock              = rock,
            rock_num          = int(round(rock_num)),
            date              = date,
            direction         = face,
            run_num           = int(round(run_num)) if not np.isnan(run_num) else np.nan,
            month             = int(round(month)),
            day               = int(round(day)),
            axis              = axis,
            burial_pct        = burial_pct,
            size_x_mm         = size_x_mm,
            size_y_mm         = size_y_mm,
            size_z_mm         = size_z_mm,
            axis1_cm          = ax1,
            axis2_cm          = ax2,
            axis3_cm          = ax3,
            h_cm              = h_cm,
            Hs_cm             = Hs_cm,
            weight_g          = weight_g,
            M_kg              = M_kg,
            density_g_cm3     = density,
            fn_peak_Hz        = fn_peak_Hz,
            fn_halfpower_Hz   = fn_halfpower_Hz,
            f1_Hz             = f1_Hz,
            f2_Hz             = f2_Hz,
            damping_ratio     = damping_ratio,
            halfpower_valid   = halfpower_valid,
            mag_gradient      = mag_gradient,
            mag_spatial_ratio = mag_spatial_ratio,
            spatial_slope     = spatial_slope,
            spatial_r2        = spatial_r2,
            spatial_grad_norm = spatial_grad_norm,  # (top-bot)/dy_mm
            log_spatial_slope = log_spatial_slope,  # log(top/bot)/dy_mm  [e-folds/mm]
            mag_top                  = mag_top,
            mag_bottom               = mag_bottom,
            y_top_mm                 = y_top_mm,
            y_bottom_mm              = y_bottom_mm,
            n_scan_points            = n_scan_points,
            # Phase features (NaN for old NPZs)
            phase_at_fn_avg_deg      = phase_at_fn_avg_deg,
            phase_spread_deg         = phase_spread_deg,
            phase_slope_deg_per_mm   = phase_slope_deg_per_mm,
            phase_flip_flag          = phase_flip_flag,
            phase_top_bot_diff_deg   = phase_top_bot_diff_deg,
            n_phase_points           = n_phase_points,
        ))

    except Exception as e:
        print(f"  ERROR ({npz_path.name}): {e}")
        skipped += 1

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)

print(f"\nWrote {len(df)} rows  ({skipped} skipped)  ->  {CSV_OUT}")
print(f"Rocks found: {sorted(df['rock'].unique())}")
print(f"Burial %:    {sorted(df['burial_pct'].unique())}")
n_spatial = df["spatial_slope"].notna().sum()
print(f"Spatial slope available:    {n_spatial}/{len(df)} rows")
if n_spatial > 0:
    print(f"  spatial_slope  mean={df['spatial_slope'].mean():.4g}  std={df['spatial_slope'].std():.4g}")
    print(f"  spatial_r2     mean={df['spatial_r2'].mean():.3f}")
    print(f"  dy_mm span     mean={(df['y_top_mm']-df['y_bottom_mm']).mean():.1f} mm")

n_phase = df["phase_spread_deg"].notna().sum()
print(f"Phase features available:   {n_phase}/{len(df)} rows")
if n_phase > 0:
    print(f"  phase_spread_deg       mean={df['phase_spread_deg'].mean():.1f}  std={df['phase_spread_deg'].std():.1f}")
    print(f"  phase_slope_deg_per_mm mean={df['phase_slope_deg_per_mm'].mean():.3f}")
    flip_count = int(df["phase_flip_flag"].sum())
    print(f"  phase_flip_flag        {flip_count}/{n_phase} measurements show a flip")
