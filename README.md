# Burial Depth Estimation for Partially Embedded Rocks Using Scanning Laser Doppler Vibrometry and Neural Networks

This repository contains the data and code for the paper:

> Y. Ruan and E. Komendera, "Burial Depth Estimation for Partially Embedded Rocks Using Scanning Laser Doppler Vibrometry and Neural Networks," submitted to *IEEE Sensors Journal*, 2026.

A geometry-informed, vibration-based pipeline estimates how deeply a rock is buried in granular soil. An instrumented hammer excites the rock in two orthogonal directions, a scanning laser Doppler vibrometer records the frequency response, and eleven physically interpretable features (cross-section width, exposed height, resonance frequency, peak mobility, half-power damping ratio, and spatial vibration decay slope, encoded as mean/difference pairs from the two strike directions) are mapped to burial depth by a Multilayer Perceptron.

**Canonical results** (strict Leave-One-Rock-Out cross-validation on 18 concrete blocks; 102 paired samples from 204 measurements; seeds 0–19; hyperparameter selection confined to training folds, with the feature-set composition supported by a training-fold feature-set selection analysis):

| Protocol | h MAPE | h RMSE |
|---|---|---|
| **Nested selection, eleven mean/difference features** (`nested_cv.py`, paper headline) | **25.1 ± 1.0%** | **0.94 ± 0.03 cm** |
| Nested selection, raw-pairs encoding (`nested_cv.py --encoding raw11`) | 27.1 ± 0.9% | 1.074 cm |
| Fixed reference configuration (16,8,4)/tanh/α=5 (`analysis/nn_LORO.py`; used for matched relative comparisons only) | 22.6 ± 0.8% | 0.881 cm |

The training-fold feature-set selection analysis (`analysis/feature_set_selector.py`) selects the combined eleven-feature set in 10 of 18 outer folds and the damping-only set in 8, never the spatial-slope-only set; the deployed configuration named by applying the inner rule to all 18 rocks is (4,2)/tanh/α=3 (`nested_cv.py --deploy`). The known-mass closed-form inversion of Jia et al. (true mass supplied, per-fold calibrated soil constant) reaches 45.9% MAPE (`analysis/baseline_jia.py`). Under the same nested protocol with the same eleven inputs, Gaussian Process reaches 39.6%, Random Forest 45.2%, and Gradient Boosting 40.6% (`analysis/nested_alt_regressors.py`); deterministic linear regression (LORO evaluation, no hyperparameter selection) reaches 31.6%. The single-measurement benchmark at the fixed configuration is 29.6% (`comparison/nn_single_dir.py`); the encoding comparison is assembled by `analysis/encoding_selector.py` (requires the meandiff11 and raw11 runs).

## Repository Structure

```
rock-vib-burial-depth/
├── README.md
├── NPZ_data/                              # 204 pre-processed NPZ files (start here)
│   ├── 05_14_r1_z_75_b_1_fn...Hz_frame_0.npz
│   └── ...
├── make_training_dataset.py               # Step 1: NPZ -> training_dataset.csv
├── paired_dataset.py                      # Shared loader (orientation-aware widths, both encodings)
├── nested_cv.py                           # Step 3: nested CV (paper headline; encodings meandiff11/raw11/meandiff9/meandiff_beta9/raw9; --deploy)
├── results/                               # Canonical artifacts backing the paper: selection tables + per-sample predictions
├── analysis/                              # Paper tables and figures (run from the repo root)
│   ├── nn_LORO.py                         #   Step 2: fixed-config ablation + figures (Table II)
│   ├── baseline_jia.py                    #   known-mass closed-form physics baseline (Table V)
│   ├── linear_baselines.py                #   deterministic linear/polynomial/dummy baselines (Table V)
│   ├── nested_alt_regressors.py           #   GP / RF / GB, eleven mean/diff inputs, nested protocol (Table V)
│   ├── feature_set_selector.py            #   feature-set selection analysis (288 candidates/fold)
│   ├── encoding_selector.py               #   nested encoding comparison (meandiff11 vs raw11)
│   ├── error_characteristics.py           #   bias, worst-case, quartile, failure-rate analysis (Table III, Fig. 5)
│   ├── plot_nested_results.py             #   regenerates paper Figs. 4-5 from nested_cv.py output
│   └── grid_search.py                     #   96-config screening search (see bias caveat in file)
├── comparison/
│   ├── nn_single_dir.py                   #   single-direction benchmark (Table II, last row)
│   ├── nn_raw_pairs.py                    #   legacy fixed-config raw-pairs exploration
│   ├── nn_deepsets.py                     #   legacy DeepSets / Siamese exploration (requires PyTorch)
│   ├── nn_transformer.py                  #   legacy Transformer exploration (requires PyTorch)
│   └── grid_search_deepsets.py            #   legacy (requires PyTorch)
└── svd_processing/                        # Optional: regenerate NPZ from raw SVD
    ├── RAW_data/                          #   204 raw SVD files (Polytec PSV-500)
    │   ├── r1_65_80_93_w960/              #   Folder: r<ID>_<x>_<y>_<z>_w<weight_g>
    │   │   ├── 05_14_r1_z_75_b_1.svd     #   File:   <MM>_<DD>_r<ID>_<axis>_<burial%>_<face>_<run>.svd
    │   │   └── ...
    │   └── ...                            #   18 rock folders
    ├── PlotAverageSpectrumFromSVD_Batch.m #   SVD -> NPZ (requires Polytec PSV software)
    └── GetPointData.m                     #   Helper for the MATLAB script
```

## Requirements

**Python** (tested with 3.10):
```
numpy
pandas
scikit-learn
matplotlib
joblib
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

**PyTorch** (optional): only required for the DeepSets/Siamese and Transformer comparisons in `comparison/`.

**MATLAB** (optional, for SVD processing only): Requires Polytec PSV software with COM/ActiveX support to read `.svd` files. Pre-processed NPZ files are included in `NPZ_data/`. Users who want to access the raw source data will need both MATLAB and Polytec PSV software installed.

## Quickstart

### Step 1: NPZ to CSV (Python)

```bash
python make_training_dataset.py
```

This reads all `.npz` files in `NPZ_data/`, extracts metadata and features, and writes `training_dataset.csv`.

### Step 2: Run LORO Evaluation (Python)

```bash
# Run the paper's proposed eleven-feature set at the fixed reference config:
python analysis/nn_LORO.py

# Run the full ablation ladder (Table II):
python analysis/nn_LORO.py --variant all

# Run only the 7-feature baseline:
python analysis/nn_LORO.py --variant baseline

# Custom hyperparameters:
python analysis/nn_LORO.py --alpha 5.0 --hidden-layers 16 8 4 --seed 0

# Skip figure generation:
python analysis/nn_LORO.py --no-plot
```

Results (CSV and figures) are saved to `output/`.

### Step 3: Nested Cross-Validation (paper headline result)

```bash
# Nested hyperparameter selection (eleven mean/difference features, paper model):
python nested_cv.py

# Raw-pairs encoding comparison under the same nested protocol:
python nested_cv.py --encoding raw11

# Feature-set candidates for the selection analysis:
python nested_cv.py --encoding meandiff9
python nested_cv.py --encoding meandiff_beta9

# Analyses assembled from the runs above (no retraining):
python analysis/feature_set_selector.py   # damping / beta / combined per fold (paper)
python analysis/encoding_selector.py      # meandiff11 vs raw11 per fold

# Named deployed configuration (inner rule on all 18 rocks):
python nested_cv.py --deploy

# Physics baseline, alternative regressors, and error analysis:
python analysis/baseline_jia.py
python analysis/linear_baselines.py
python analysis/nested_alt_regressors.py
python analysis/error_characteristics.py     # requires nested_cv.py output
python analysis/plot_nested_results.py       # paper Figs. 4-5; requires nested_cv.py output
```

Note: `nested_cv.py` refits 96 configurations inside every LORO fold and takes
roughly an hour on a 16-core machine; per-fold selections are checkpointed to
`output/` and the script resumes if interrupted.

### Optional: Regenerate NPZ from Raw SVD (MATLAB)

If you have Polytec PSV software installed, you can regenerate the NPZ files from the raw SVD scans:

```matlab
cd svd_processing
PlotAverageSpectrumFromSVD_Batch
```

This reads all `.svd` files under `svd_processing/RAW_data/` and writes `.npz` + `.png` files into `NPZ_data/`. The raw SVD files contain per-point frequency response data (H1 velocity/force, magnitude and phase) that can also be accessed directly using `GetPointData.m` for other signal types (e.g., displacement, real/imaginary components).

## Concrete Block Specifications

| Block | *x* (mm) | *y* (mm) | *z* (mm) | Weight (g) | Density (g/cm³) | Burial levels |
|-------|----------|----------|----------|------------|-----------------|---------------|
| r1    | 65.0     | 80.0     | 93.0     | 960        | 1.99            | 25--75%       |
| r2    | 80.0     | 93.0     | 128.0    | 1907       | 2.00            | 25--50%       |
| r3    | 49.4     | 68.0     | 93.0     | 634        | 2.03            | 20--60%       |
| r4    | 93.0     | 110.0    | 133.0    | 2749       | 2.02            | 20--60%       |
| r5    | 68.5     | 93.0     | 108.0    | 1381       | 2.01            | 20--60%       |
| r6    | 32.0     | 69.0     | 93.0     | 412        | 2.01            | 50--75%       |
| r7    | 59.0     | 93.3     | 109.0    | 1170       | 1.95            | 20--60%       |
| r8    | 63.0     | 74.3     | 80.9     | 748        | 1.98            | 20--60%       |
| r9    | 50.0     | 67.3     | 74.7     | 500        | 1.99            | 20--60%       |
| r11   | 80.4     | 110.0    | 140.0    | 2442       | 1.97            | 15--60%       |
| r12   | 80.0     | 98.0     | 121.0    | 1907       | 2.01            | 25--75%       |
| r13   | 82.0     | 92.4     | 101.6    | 1488       | 1.93            | 30--60%       |
| r14   | 75.0     | 105.0    | 130.0    | 1961       | 1.92            | 20--60%       |
| r15   | 62.6     | 91.4     | 122.5    | 1347       | 1.92            | 20--75%       |
| r16   | 52.0     | 63.4     | 69.2     | 445        | 1.95            | 25--50%       |
| r17   | 49.0     | 81.0     | 110.0    | 874        | 2.00            | 25--75%       |
| r18   | 69.0     | 92.5     | 94.7     | 1185       | 1.96            | 25--75%       |
| r19   | 75.5     | 92.3     | 100.8    | 1353       | 1.93            | 25--75%       |

## Data Format

### SVD Files
Binary Polytec PSV-500 scan files containing per-point H1 frequency response functions (velocity/force) with magnitude and phase.

### Folder Naming Convention
`r<ID>_<x_mm>_<y_mm>_<z_mm>_w<weight_g>` encodes rock dimensions (mm) and mass (g).

### File Naming Convention
`<MM>_<DD>_r<ID>_<axis>_<burial%>_<face>_<run>.svd`
- `axis`: `x` (short dimension vertical) or `z` (tall dimension vertical)
- `burial%`: percentage of vertical dimension buried (e.g., 25, 40, 60)
- `face`: `b` or `s` (two perpendicular hammer strike directions)
- `run`: replicate number

## Citation

```bibtex
@article{Ruan_Komendera_2026,
  author  = {Ruan, Yiyan and Komendera, Erik},
  title   = {Burial Depth Estimation for Partially Embedded Rocks Using
             Scanning Laser Doppler Vibrometry and Neural Networks},
  journal = {IEEE Sensors Journal},
  year    = {2026},
  note    = {submitted}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
