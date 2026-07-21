# Inverse-model closed-loop FEM validation

This focused check uses the existing QA-passed synthetic CSV and saved MLP. It generates no data, fits no model, and runs no calibration or mesh sweep.

## Method

- Selected samples: **50**, from **saved test split IDs for mlp**.
- Input to inverse model: the original three polar-angle-sorted minima in metres.
- Model output: eight raw displacement coordinates in Wolfram W1--W4 order.
- No prediction clipping is applied.
- FEM transform: `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.
- Forward model: real-scale, all-positive electrodes, fixed grounded 50 mm outer circle, robust minima mode, practical 500 µm central mesh.
- Recomputed and original minima are compared by minimum-total-distance Hungarian assignment.
- Aggregate errors include only `status=ok` rows: exactly three robust-accepted minima and pairwise separation at least 0.15 mm.

## Results

| Metric | Value |
|---|---:|
| Mean closed-loop minimum error | 124.874099 µm |
| Median closed-loop minimum error | 102.119261 µm |
| 95th-percentile error | 306.938531 µm |
| Maximum error | 583.981970 µm |
| Exactly-three robust topology | 50 / 50 |
| Included clean rows | 50 / 50 |
| Solver failures | 0 |
| Ambiguous/rejected rows | 0 |
| Rows with robust-rejected extra candidates | 3 |
| Selected interpolation-sensitive rows | 0 |

## First-demo assessment

All 50 selected cases preserved the clean exactly-three topology, and the mean and 95th-percentile loop-closure errors are sub-millimetre. This makes the saved model physically useful for a coarse first demonstration. It is not a precision inverse: the worst matched minimum error is **583.981970 µm**, and the test does not establish unique recovery of the original eight displacement coordinates.

## Prediction-range audit

The MLP produced **4** coordinates outside the generator's ±500 µm training range. The largest absolute predicted coordinate was **569.571862 µm**. These values were not clipped before FEM evaluation.

## Interpretation boundary

This experiment evaluates physical loop closure in minimum-position space, not recovery of the unique original electrode displacements. Six minimum coordinates cannot generally identify eight independently sampled displacement coordinates uniquely. The CSV preserves every topology rejection, solver failure, assignment, and candidate diagnostic used by this summary.
