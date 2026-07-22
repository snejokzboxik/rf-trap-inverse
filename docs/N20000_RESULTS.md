# N=20000 inverse reconstruction results

This report records the N=20000 synthetic-data inverse-reconstruction
milestone. It is a validation against the same FEM forward model used to make
the synthetic dataset; it is not yet independent Wolfram or experimental
validation.

## Main result

The N=20000 MLP inverse model achieved a held-out electrode-displacement test
MAE of **104.82 um**. Closed-loop FEM validation on 100 samples gave a mean
matched-minimum position error of **96.30 um**, median **79.38 um**, p95
**212.09 um**, and maximum **403.64 um**, with **100/100** valid
three-minima topologies, zero solver failures, and zero ambiguous-branch
rejections.

Here, **p95** is the error below which 95% of all Hungarian-matched minimum
position errors in the selected validation set fall.

## Dataset generation and QA

| Item | Result |
|---|---:|
| Requested / completed attempts | 20000 / 20000 |
| Clean / rejected rows | 20000 / 0 |
| Ambiguous branches / solver failures | 0 / 0 |
| Generation runtime | 19567.931 s |
| Interrupted | False |
| QA `ml_ready` | True |
| QA critical issues | 0 |
| QA polar-order violations | 0 |

The dataset is at `validation_results/generated_dataset_20000`; its QA report
and summary statistics are at
`validation_results/generated_dataset_20000_qa/qa_report.md` and
`validation_results/generated_dataset_20000_qa/summary_stats.csv`.

The real-scale forward configuration retains a 500 um central mesh and all
electrode potentials at +1 V. Raw displacement pairs use Wolfram order
W1 upper-right, W2 lower-right, W3 upper-left, W4 lower-left. FEM uses
F1 upper-left, F2 upper-right, F3 lower-left, F4 lower-right, with the
canonical absolute-displacement transform `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.

## Inverse training

The N=20000 train/test split contains 16000 / 4000 samples. The best saved
model is the MLP in `validation_results/inverse_model_20000`.

| Metric | N=5000 MLP | N=20000 MLP |
|---|---:|---:|
| Test MAE (um) | 107.140452 | 104.823850 |
| Test RMSE (um) | 137.553139 | 134.553068 |
| Maximum absolute coordinate error (um) | 508.514970 | 519.119544 |

N=20000 improves held-out MAE and RMSE. The maximum coordinate error is
slightly higher, so this metric is retained as a caveat rather than presented
as a uniform improvement.

## Closed-loop FEM validation

The closed loop is: true minima -> MLP-predicted Wolfram displacements ->
`-[W3,W1,W4,W2]` -> robust forward FEM -> Hungarian matching against the
original minima. All errors below are in micrometres.

| Metric | N=5000, N=50 | N=20000, N=50 | N=20000, N=100 |
|---|---:|---:|---:|
| Mean error | 124.874099 | 93.430449 | 96.296420 |
| Median error | 102.119261 | 75.775161 | 79.384966 |
| p95 error | 306.938531 | 209.820318 | 212.087844 |
| Maximum error | 583.981970 | 403.643453 | 403.643453 |
| Exactly-three topology | 50 / 50 | 50 / 50 | 100 / 100 |
| Solver failures | 0 | 0 | 0 |
| Ambiguous/rejected rows | 0 | 0 | 0 |

The N=50 and N=100 summaries are saved under
`validation_results/closed_loop_inverse_20000_n50` and
`validation_results/closed_loop_inverse_20000_n100`, respectively.

## Interpretation and caveats

N=20000 gives stable closed-loop reconstruction with mean minima-position
error below 100 um on the 100-sample validation subset. This improves the
N=5000 N=50 mean, median, p95, and maximum closed-loop errors while preserving
the exactly-three-minima topology in every selected case.

- N=50 and N=100 remain limited deterministic subsets, not an exhaustive test
  of the eight-dimensional displacement space.
- The MLP can predict slightly outside the sampled +/-500 um displacement
  range; the N=100 check had three such coordinates before FEM evaluation.
- The reconstruction is trained on synthetic FEM data. Independent Wolfram and
  experimental validation remains necessary.
- `random_forest.joblib` is intentionally not tracked: it is large and is not
  the best model.
