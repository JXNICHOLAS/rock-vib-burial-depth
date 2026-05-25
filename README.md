# Burial Depth Estimation for Partially Embedded Rocks Using Scanning Laser Doppler Vibrometry and Neural Networks

This repository contains data and code for the paper:

> Y. Ruan and E. Komendera, "Burial Depth Estimation for Partially Embedded Rocks Using Scanning Laser Doppler Vibrometry and Neural Networks," submitted to *IEEE Sensors Journal*, 2026.

A vibration-based pipeline estimates how deeply a rock is buried in granular soil. An instrumented hammer excites the rock, a scanning laser Doppler vibrometer records the frequency response, and eleven physically interpretable features (resonance frequency, peak mobility, half-power damping ratio, spatial vibration decay, exposed height, and cross-section width, encoded as mean/difference pairs from two orthogonal strikes) are mapped to burial depth by a Multilayer Perceptron. Under strict Leave-One-Rock-Out cross-validation on 16 concrete blocks (90 paired samples from 180 measurements), the model achieves 24.4% MAPE and 1.02 cm RMSE.

## Repository Structure

```
rock-vib-burial-depth/
├── README.md
├── NPZ_data/                              # 180 pre-processed NPZ files (start here)
│   ├── 05_14_r1_z_75_b_1_fn...Hz_frame_0.npz
│   └── ...
├── make_training_dataset.py               # Step 1: NPZ -> training_dataset.csv
├── nn_LORO.py                             # Step 2: LORO evaluation + figure generation
└── svd_processing/                        # Optional: regenerate NPZ from raw SVD
    ├── RAW_data/                          #   180 raw SVD files (Polytec PSV-500)
    │   ├── r1_65_80_93_w960/              #   Folder: r<ID>_<x>_<y>_<z>_w<weight_g>
    │   │   ├── 05_14_r1_z_75_b_1.svd     #   File:   <MM>_<DD>_r<ID>_<axis>_<burial%>_<face>_<run>.svd
    │   │   └── ...
    │   └── ...                            #   16 rock folders
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
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib
```

**MATLAB** (optional, for SVD processing only): Requires Polytec PSV software with COM/ActiveX support to read `.svd` files. Pre-processed NPZ files are included in `NPZ_data/`. Users who want to access the raw source data will need both MATLAB and Polytec PSV software installed.

## Quickstart

### Step 1: NPZ to CSV (Python)

```bash
python make_training_dataset.py
```

This reads all `.npz` files in `NPZ_data/`, extracts metadata and features, and writes `training_dataset.csv`.

### Step 2: Run LORO Evaluation (Python)

```bash
# Run the paper's 11-feature model (default):
python nn_LORO.py

# Run all feature-set variants for comparison:
python nn_LORO.py --variant all

# Run only the 7-feature baseline:
python nn_LORO.py --variant baseline

# Custom hyperparameters:
python nn_LORO.py --alpha 5.0 --hidden-layers 16 8 --seed 0

# Skip figure generation:
python nn_LORO.py --no-plot
```

Results (CSV and figures) are saved to `output/`.

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
