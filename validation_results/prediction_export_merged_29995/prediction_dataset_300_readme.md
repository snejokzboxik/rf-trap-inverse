# Prediction export from the merged N=29995 inverse model

This file contains **300** deterministic rows selected with random state `20260725`. It is saved-model inference only: no new FEM solve, synthetic generation, calibration, or training was run.
The rows are sampled from the full source dataset and are not guaranteed to belong to the model's held-out test split; the error summary is descriptive.

## Column blocks

1. `sample_id`: original row ID in the source ML dataset.
2. `true_w*_d*_m`: eight true electrode-displacement coordinates in metres.
3. `min*_x_m`, `min*_y_m`: six equilibrium/minimum-position coordinates in metres.
4. `pred_w*_d*_m`: eight inverse-model predicted displacements in metres.
5. `true_*_um` and `pred_*_um`: readable micrometre copies of the displacements.
6. `error_*_um`: signed prediction error, predicted minus true, in micrometres.
7. `w*_vector_error_um`: Euclidean `(dx,dy)` error for each electrode.
8. `row_mae_um`: mean absolute error across all eight displacement coordinates.

## Convention and provenance

Raw displacement columns use Wolfram electrode order: W1 upper-right, W2 lower-right, W3 upper-left, W4 lower-left. The project transform to internal FEM order remains `[-W3, -W1, -W4, -W2]`.

- Source dataset: `validation_results\generated_dataset_merged_29995\synthetic_clean_ml.csv`
- Saved model: `validation_results\inverse_model_merged_29995\mlp.joblib`
- Metre-valued columns end in `_m`; micrometre-valued columns end in `_um`.

## Inference summary

- Coordinate MAE: **101.637934 µm**
- Coordinate RMSE: **128.733378 µm**
- Maximum absolute coordinate error: **434.347688 µm**
- Mean electrode-vector error: **160.316527 µm**
- Mean row MAE: **101.637934 µm**
