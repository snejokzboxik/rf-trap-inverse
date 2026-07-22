# Prediction export convention

Prediction exports are inference-only tables for sharing inverse-model results.
They do not run FEM, generate synthetic samples, calibrate physics, or retrain a
model.

## Standard model choice

- **Latest/largest export:**
  `validation_results/prediction_export_merged_51974/`. Use this by default
  because merged N=51974 is the latest/largest training pipeline and has the
  best ordinary held-out regression MAE.
- **Closed-loop-best export:**
  `validation_results/prediction_export_merged_29995/`. Use this when the best
  observed closed-loop headline model is specifically requested.

The N=29995 and N=51974 closed-loop headline runs used different deterministic
subsets, so their reported closed-loop summaries are not a paired model
comparison.

## CSV layout

Each row contains these blocks in Wolfram electrode order:

1. Original `sample_id`.
2. Eight true displacement coordinates in metres.
3. Six coordinates for three polar-angle-sorted equilibrium/minimum positions
   in metres.
4. Eight saved-model predicted displacement coordinates in metres.
5. True and predicted displacement copies in micrometres.
6. Eight signed coordinate errors in micrometres, defined as predicted minus
   true.
7. Four per-electrode `(dx,dy)` vector errors in micrometres.
8. Row-level mean absolute coordinate error in micrometres.

Wolfram order is W1 upper-right, W2 lower-right, W3 upper-left, W4 lower-left.
The project transform to FEM order remains `[-W3, -W1, -W4, -W2]`.

Rows are sampled deterministically without replacement from the full source ML
dataset. They are not guaranteed to belong to the model's held-out test split,
so export error summaries are descriptive rather than independent test metrics.

## Default command

```powershell
python -m rf_trap_forward.export_prediction_dataset --n 300 --random-state 20260725
```

The no-argument paths now select the merged N=51974 dataset, MLP, and output
directory. Pass explicit `--dataset`, `--model`, and `--output-csv` paths to
create a closed-loop-best N=29995 export instead.
