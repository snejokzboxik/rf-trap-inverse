# Baseline inverse-model experiment

This experiment uses only `validation_results/generated_dataset/synthetic_clean.csv`. It performs no FEM solve, calibration, mesh sweep, or new data generation.

## Data and convention

- Samples: 20000 (16000 train, 4000 test).
- Split: `test_size=0.2`, `random_state=42`.
- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.
- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.
- Internal fit units are metres; every reported error metric is in micrometres.
- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.

The inverse is intrinsically underdetermined at the coordinate level: six observed minimum coordinates are being used to recover eight independently sampled displacement coordinates. These baselines therefore measure useful predictive correlation; they do not establish a unique physical inverse.

## Held-out metrics

| Model | MAE (µm) | RMSE (µm) | Max absolute (µm) | Mean vector error (µm) | Max vector error (µm) | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 104.823850 | 134.553068 | 519.119544 | 166.431278 | 559.712292 | 40.147 |
| random_forest | 138.523075 | 170.529654 | 647.479309 | 216.624358 | 695.890643 | 3.529 |
| ridge | 171.875916 | 210.730841 | 1109.270453 | 265.518673 | 1156.355162 | 0.061 |

Best model by test MAE: **mlp** at **104.823850 µm**.
For context, predicting the eight training-set coordinate means for every test row gives MAE **250.143124 µm**, RMSE **288.941067 µm**, and maximum absolute error **503.884136 µm**.

## Interpretation

The best learned model reduces coordinate MAE by **58.09%** relative to the train-mean predictor. This is useful for a coarse first demo and confirms that the minima encode substantial displacement information. However, the **104.82 µm** MAE and **519.12 µm** worst coordinate error are not precision-control accuracy. The underdetermined 6-to-8 mapping also prevents a unique-inverse claim.

`per_output_metrics.csv` contains all eight coordinate metrics and all four electrode-vector mean/max errors. `test_predictions.csv` contains one held-out prediction per model and sample. Positive coordinate error means predicted minus true.

## Saved models and plots

- `ridge.joblib`, `random_forest.joblib`, and `mlp.joblib` include preprocessing where applicable.
- `plots/predicted_vs_true.png`
- `plots/coordinate_error_histogram.png`
- `plots/electrode_vector_error.png`

A consumer must supply minimum coordinates in the same metre-valued, polar-angle-sorted order used by the generated dataset. Joblib files should only be loaded from this trusted project output.
