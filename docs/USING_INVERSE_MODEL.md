# Using the trained inverse model

The prediction interface loads an existing trusted MLP artifact and predicts
four electrode-displacement vectors from three pseudopotential minima. It does
not run FEM, retrain a model, generate data, or change the physical model.

## Model choice

The default model is:

`validation_results/inverse_model_merged_51974/mlp.joblib`

Merged N=51974 is the latest/largest trained pipeline and has the best ordinary
held-out regression MAE, so it is the normal prediction default. The alternate
model at `validation_results/inverse_model_merged_29995/mlp.joblib` retains the
best observed closed-loop headline metric. Select it explicitly when a
closed-loop-best prediction is requested.

Only load joblib artifacts from this trusted project; joblib deserialization is
not safe for untrusted files.

## Scientific input and units

The six inputs are three deterministic, polar-angle-sorted minimum positions:

`min1_x_m,min1_y_m,min2_x_m,min2_y_m,min3_x_m,min3_y_m`

### Canonical minima ordering

The ordering is inherited from the existing forward/minima and synthetic-data
pipeline, not chosen by the prediction interface: points are sorted by
increasing `atan2(y, x)` after wrapping the angle into `[0, 2*pi)`. This is the
same `sort_points_by_polar_angle` helper used when writing synthetic minima and
when ordering recomputed minima in closed-loop validation. Direct, CSV, and GUI
inputs are auto-sorted by this rule before prediction by default. The terminal
output and GUI show the ordered coordinates used by the model.

Use `--no-sort-minima` on the CLI, or clear “Auto-sort minima before prediction”
in the GUI, only when the input is already known to be in the training order.
Arbitrary manual ordering changes the six-feature vector and can therefore
change the predicted displacements.

The CLI accepts direct values in metres (`--units m`) or millimetres
(`--units mm`) and always converts to metres before inference. CSV inputs use
the same canonical six-column header. Normally their values are metres; pass
`--units mm` only when the numeric values in that canonical table are actually
millimetres.

## Electrode order and output

Predictions use raw Wolfram electrode order:

- W1 = upper-right
- W2 = lower-right
- W3 = upper-left
- W4 = lower-left

The CLI and CSV also report the internal FEM-order transform:

`F1,F2,F3,F4 = -[W3,W1,W4,W2]`

CSV output contains the six input minima in metres, predicted Wolfram-order
displacements in metres and micrometres, FEM-order transformed displacements in
metres and micrometres, and one warning column. A warning is emitted if any
predicted Wolfram coordinate exceeds the training range of ±500 µm.

## Direct CLI prediction

Metres:

```powershell
python -m rf_trap_forward.predict_inverse --minima "-0.001596,0.003869;-0.001836,-0.003034;0.004218,-0.001076" --units m
```

Millimetres:

```powershell
python -m rf_trap_forward.predict_inverse --minima "-1.596,3.869;-1.836,-3.034;4.218,-1.076" --units mm
```

To inspect the effect of preserving the supplied order, add
`--no-sort-minima`.

Choose the closed-loop-best N=29995 model explicitly with:

```powershell
python -m rf_trap_forward.predict_inverse --model validation_results/inverse_model_merged_29995/mlp.joblib --minima "-1.596,3.869;-1.836,-3.034;4.218,-1.076" --units mm
```

## CSV prediction

```powershell
python -m rf_trap_forward.predict_inverse --input-csv examples/example_minima_input.csv --output-csv validation_results/manual_predictions/predicted_displacements.csv --units m
```

The output directory is created automatically. Direct input can also be saved
by adding `--output-csv PATH`.

## Tkinter desktop GUI

From the repository root:

```powershell
python app_inverse_model_tk.py
```

The GUI uses N=51974 by default, permits selecting another `.joblib` file,
accepts six direct coordinates in metres or millimetres, loads batch CSV input,
auto-sorts minima with the canonical pipeline rule, displays Wolfram and FEM
vectors in micrometres, and saves the current single or batch prediction to CSV.
The output panel is selectable, supports Ctrl+C, and has a Copy output button.
Loading a CSV predicts all rows and displays the first five rows in the text
panel.

## Limitations

- The model was trained on synthetic outputs from the documented FEM pipeline;
  it is not a substitute for independent Wolfram or experimental validation.
- Training electrode-displacement coordinates span ±500 µm. Predictions beyond
  this interval are extrapolations and are explicitly warned.
- Three minima provide six observed coordinates for eight displacement outputs;
  the inverse is not guaranteed to be unique.
- Polar-angle input ordering must match the training convention. Reordering the
  same three physical minima can change the model input and prediction.
