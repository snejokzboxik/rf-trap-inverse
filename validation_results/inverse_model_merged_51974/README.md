# Baseline inverse-model experiment

This experiment uses only `validation_results/generated_dataset/synthetic_clean.csv`. It performs no FEM solve, calibration, mesh sweep, or new data generation.

## Data and convention

- Samples: 51974 (41579 train, 10395 test).
- Split: `test_size=0.2`, `random_state=42`.
- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.
- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.
- Internal fit units are metres; every reported error metric is in micrometres.
- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.

The inverse is intrinsically underdetermined at the coordinate level: six observed minimum coordinates are being used to recover eight independently sampled displacement coordinates. These baselines therefore measure useful predictive correlation; they do not establish a unique physical inverse.

## Held-out metrics

| Model | MAE (µm) | RMSE (µm) | Max absolute (µm) | Mean vector error (µm) | Max vector error (µm) | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 102.891015 | 132.217767 | 530.827205 | 163.537347 | 564.401813 | 111.476 |
| random_forest | 129.513125 | 160.499889 | 638.749610 | 202.861061 | 684.027777 | 13.003 |
| ridge | 170.616817 | 209.663012 | 985.042242 | 264.115242 | 1083.688821 | 0.028 |

Best model by test MAE: **mlp** at **102.891015 µm**.
For context, predicting the eight training-set coordinate means for every test row gives MAE **250.174438 µm**, RMSE **288.807303 µm**, and maximum absolute error **502.632353 µm**.

## Interpretation

The best learned model reduces coordinate MAE by **58.87%** relative to the train-mean predictor. This is useful for a coarse first demo and confirms that the minima encode substantial displacement information. However, the **102.89 µm** MAE and **530.83 µm** worst coordinate error are not precision-control accuracy. The underdetermined 6-to-8 mapping also prevents a unique-inverse claim.

`per_output_metrics.csv` contains all eight coordinate metrics and all four electrode-vector mean/max errors. `test_predictions.csv` contains one held-out prediction per model and sample. Positive coordinate error means predicted minus true.

## Saved models and plots

- Saved model artifacts: `ridge.joblib`, `mlp.joblib`. Metrics still include every fitted model.
- `plots/predicted_vs_true.png`
- `plots/coordinate_error_histogram.png`
- `plots/electrode_vector_error.png`

A consumer must supply minimum coordinates in the same metre-valued, polar-angle-sorted order used by the generated dataset. Joblib files should only be loaded from this trusted project output.
