# Baseline inverse-model experiment: merged N=29995

This experiment uses only
`validation_results/generated_dataset_merged_29995/synthetic_clean_ml.csv`.
It performs no FEM solve, calibration, mesh sweep, or new data generation.

## Data and convention

- Samples: 29995 (23996 train, 5999 test).
- Split: `test_size=0.2`, `random_state=42`.
- Input: three deterministic polar-angle-sorted minimum positions, shape `(N, 6)`.
- Target: four raw displacement vectors in user-facing Wolfram order, shape `(N, 8)`.
- Internal fit units are metres; every reported error metric is in micrometres.
- Ridge and MLP standardize X and y. Random forest uses raw metre-valued features and targets.

The inverse is intrinsically underdetermined at the coordinate level: six
observed minimum coordinates are used to recover eight independently sampled
displacement coordinates. These baselines measure predictive correlation; they
do not establish a unique physical inverse.

## Held-out metrics

| Model | MAE (um) | RMSE (um) | Max absolute (um) | Mean vector error (um) | Max vector error (um) | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 103.792467 | 132.485981 | 542.911477 | 164.249256 | 549.760100 | 85.397 |
| random_forest | 135.384339 | 167.050567 | 643.325010 | 211.489540 | 691.615000 | 7.265 |
| ridge | 169.879860 | 208.567773 | 1100.177430 | 262.820123 | 1145.173969 | 0.023 |

Best model by test MAE: **mlp** at **103.792467 um**. The train-mean predictor
has MAE 248.788806 um, so the MLP reduces coordinate MAE by 58.28% relative to
that baseline.

## Saved outputs

- `mlp.joblib` and `ridge.joblib` are retained.
- `random_forest.joblib` is intentionally omitted because it is roughly
  1.17 GB and random forest is not the best model.
- `per_output_metrics.csv` and `test_predictions.csv` contain the detailed
  coordinate results.
- Plots are under `plots/`.

A consumer must provide minima in the same metre-valued, polar-angle-sorted
order. Joblib files should be loaded only from this trusted project output.
