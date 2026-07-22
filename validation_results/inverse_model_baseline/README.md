# Baseline inverse-model experiment

This experiment uses only `validation_results/generated_dataset/synthetic_clean.csv`. It performs no FEM solve, calibration, mesh sweep, or new data generation.

## Data and convention

- Samples: 1000 (800 train, 200 test).
- Split: `test_size=0.2`, `random_state=42`.
- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.
- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.
- Internal fit units are metres; every reported error metric is in micrometres.
- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.

The inverse is intrinsically underdetermined at the coordinate level: six observed minimum coordinates are being used to recover eight independently sampled displacement coordinates. These baselines therefore measure useful predictive correlation; they do not establish a unique physical inverse.

## Held-out metrics

| Model | MAE (µm) | RMSE (µm) | Max absolute (µm) | Mean vector error (µm) | Max vector error (µm) | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 119.154312 | 149.465719 | 439.328131 | 187.644343 | 454.410323 | 1.468 |
| random_forest | 167.207644 | 202.027733 | 574.062190 | 260.587513 | 640.859217 | 0.389 |
| ridge | 168.546478 | 206.019227 | 641.155772 | 260.462529 | 790.319352 | 0.004 |

Best model by test MAE: **mlp** at **119.154312 µm**.
For context, predicting the eight training-set coordinate means for every test row gives MAE **250.025334 µm**, RMSE **288.430356 µm**, and maximum absolute error **515.430317 µm**.

## Interpretation

The best learned model reduces coordinate MAE by **52.34%** relative to the train-mean predictor. This is useful for a coarse first demo and confirms that the minima encode substantial displacement information. However, the **119.15 µm** MAE and **439.33 µm** worst coordinate error are not precision-control accuracy. The underdetermined 6-to-8 mapping also prevents a unique-inverse claim.

`per_output_metrics.csv` contains all eight coordinate metrics and all four electrode-vector mean/max errors. `test_predictions.csv` contains one held-out prediction per model and sample. Positive coordinate error means predicted minus true.

## Saved models and plots

- `ridge.joblib` and `mlp.joblib` include preprocessing where applicable.
- Random-forest metrics remain in the CSV reports, but
  `random_forest.joblib` is intentionally omitted because random forest was
  not the best model and its artifact was unnecessarily large.
- `plots/predicted_vs_true.png`
- `plots/coordinate_error_histogram.png`
- `plots/electrode_vector_error.png`

A consumer must supply minimum coordinates in the same metre-valued, polar-angle-sorted order used by the generated dataset. Joblib files should only be loaded from this trusted project output.
