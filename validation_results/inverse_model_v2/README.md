# Improved inverse-model comparison (v2)

This experiment reads only the existing QA-passed N=1000 clean CSV. It runs no FEM solve, data generation, closed-loop validation, calibration, or mesh sweep.

## Evaluation design

- Primary split: `test_size=0.2`, `random_state=42`; 800 train and 200 test rows.
- Repeated splits: 5 seeds: `42, 43, 44, 45, 46`.
- X: six polar-angle-sorted minimum coordinates in metres.
- y: eight raw W1--W4 displacement coordinates in Wolfram order, in metres.
- Reported errors are in micrometres.
- Raw predictions and coordinate-wise clipping to ±500 µm are evaluated separately. The saved model includes clipping only when the clipped variant wins.

## Primary-split metrics

| Model | Variant | MAE (µm) | RMSE (µm) | Max absolute (µm) | Outside before/after | Fit time (s) |
|---|---|---:|---:|---:|---:|---:|
| tuned_mlp | raw | 117.850238 | 148.982013 | 453.396487 | 25/25 | 9.146 |
| tuned_mlp | clipped | 117.260924 | 148.666133 | 453.396487 | 25/0 | 9.146 |
| extra_trees | raw | 162.938660 | 198.579814 | 594.512376 | 0/0 | 0.712 |
| extra_trees | clipped | 162.938660 | 198.579814 | 594.512376 | 0/0 | 0.712 |
| hist_gradient_boosting | raw | 145.556474 | 181.901449 | 660.703263 | 3/3 | 5.595 |
| hist_gradient_boosting | clipped | 145.538019 | 181.893133 | 660.703263 | 3/0 | 5.595 |
| knn | raw | 169.492030 | 206.745283 | 635.517822 | 0/0 | 0.002 |
| knn | clipped | 169.492030 | 206.745283 | 635.517822 | 0/0 | 0.002 |

## Five-split stability (clipped variants)

| Model | Mean MAE (µm) | Std (µm) | Minimum (µm) | Maximum (µm) |
|---|---:|---:|---:|---:|
| tuned_mlp | 114.468240 | 3.282773 | 109.168352 | 117.395197 |
| extra_trees | 160.476663 | 1.702336 | 158.614551 | 162.938660 |
| hist_gradient_boosting | 141.133331 | 2.298094 | 138.958429 | 145.538019 |
| knn | 167.050333 | 1.844383 | 163.815590 | 169.492030 |

## Best model and clipping

Best primary-split result: **tuned_mlp (clipped)** with MAE **117.260924 µm**, RMSE **148.666133 µm**, and maximum absolute error **453.396487 µm**.
The first baseline MLP MAE was **119.154312 µm**; v2 changes it by **1.893388 µm** (1.59% improvement when positive).
The first baseline maximum error was **439.328131 µm**; the winning v2 result changes it by **-14.068356 µm** (negative means v2 is worse).

Clipping cannot increase coordinate-wise absolute error because every target lies within ±500 µm. It does not improve primary-split maximum error for any tested model.

## Recommendation

Keep the tuned clipped MLP as a v2 candidate, but do not replace the previous baseline yet. Its MAE gain is small and its primary-split maximum error is worse; the requested physical closed-loop comparison has also deliberately not been run in this experiment.

This remains an underdetermined six-observation/eight-target inverse. Better held-out regression does not prove unique recovery of the physical electrode displacements. Run a separate closed-loop FEM validation before treating the replacement as physically superior.

## Files

- `metrics.csv`: primary raw/clipped model comparison.
- `repeated_split_metrics.csv`: all five split/model/variant results.
- `per_output_metrics.csv`: eight coordinate and four electrode-vector metrics.
- `test_predictions.csv`: primary held-out predictions and errors.
- `best_model.joblib`: winning estimator, including clipping when selected.
- `plots/`: four requested diagnostic figures.
