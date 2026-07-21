# Baseline inverse-model experiment

This experiment uses only `validation_results/generated_dataset/synthetic_clean.csv`. It performs no FEM solve, calibration, mesh sweep, or new data generation.

## Data and convention

- Samples: 5000 (4000 train, 1000 test).
- Split: `test_size=0.2`, `random_state=42`.
- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.
- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.
- Internal fit units are metres; every reported error metric is in micrometres.
- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.

The inverse is intrinsically underdetermined at the coordinate level: six observed minimum coordinates are being used to recover eight independently sampled displacement coordinates. These baselines therefore measure useful predictive correlation; they do not establish a unique physical inverse.

## Held-out metrics

| Model | MAE (µm) | RMSE (µm) | Max absolute (µm) | Mean vector error (µm) | Max vector error (µm) | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 107.140452 | 137.553139 | 508.514970 | 169.898856 | 534.464448 | 13.477 |
| random_forest | 149.843583 | 183.843074 | 668.584029 | 233.470636 | 705.553784 | 0.875 |
| ridge | 170.689657 | 209.134493 | 883.233835 | 263.385692 | 975.749887 | 0.007 |

Best model by test MAE: **mlp** at **107.140452 µm**.
For context, predicting the eight training-set coordinate means for every test row gives MAE **248.728201 µm**, RMSE **288.128496 µm**, and maximum absolute error **505.207419 µm**.

## Interpretation

The best learned model reduces coordinate MAE by **56.92%** relative to the train-mean predictor. This is useful for a coarse first demo and confirms that the minima encode substantial displacement information. However, the **107.14 µm** MAE and **508.51 µm** worst coordinate error are not precision-control accuracy. The underdetermined 6-to-8 mapping also prevents a unique-inverse claim.

`per_output_metrics.csv` contains all eight coordinate metrics and all four electrode-vector mean/max errors. `test_predictions.csv` contains one held-out prediction per model and sample. Positive coordinate error means predicted minus true.

## Saved models and plots

- `ridge.joblib`, `random_forest.joblib`, and `mlp.joblib` include preprocessing where applicable.
- `plots/predicted_vs_true.png`
- `plots/coordinate_error_histogram.png`
- `plots/electrode_vector_error.png`

A consumer must supply minimum coordinates in the same metre-valued, polar-angle-sorted order used by the generated dataset. Joblib files should only be loaded from this trusted project output.
