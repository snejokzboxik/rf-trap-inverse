# N=5000 synthetic-data results

This report records the completed N=5000 synthetic dataset, its QA and inverse
training results, and small physical closed-loop checks. It does not claim an
independent experimental validation.

For the subsequent N=20000 scale-up and its N=100 closed-loop check, see
[N20000_RESULTS.md](N20000_RESULTS.md).

## Dataset generation

- Requested and clean samples: **5000**; rejected samples: **0**.
- Seed: **20260721**.
- Each Wolfram-order displacement coordinate was sampled uniformly over
  **[-500, +500] um**.
- FEM configuration: real-scale geometry with a **500 um central mesh**;
  all four electrodes at **+1 V** and the fixed outer boundary at 0 V.
- Wolfram-to-FEM displacement convention:
  `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.

## Dataset QA

The read-only QA result is `ml_ready=True`.

| Check | Result |
|---|---:|
| Clean rows | 5000 |
| Rejected rows | 0 |
| Critical issues | 0 |
| Polar-order violations | 0 |

The clean table has finite values, no duplicate rows/IDs/inputs, displacement
coordinates inside the declared bounds, exact stored Wolfram-to-FEM transforms,
and minimum pairwise separations above the 0.15 mm clean-data threshold.

## Inverse training

The best saved N=5000 inverse model is the **MLP**. Its held-out test metrics
are reported in micrometres.

| Metric | Value |
|---|---:|
| Test MAE | 107.140452 um |
| Test RMSE | 137.553139 um |
| Maximum absolute coordinate error | 508.514970 um |

## Closed-loop validation

Both validations feed true minima into the saved MLP, transform its predicted
Wolfram-order displacements with `-[W3,W1,W4,W2]`, rerun the robust forward FEM
solver, and Hungarian-match recomputed minima to the original minima.

| Metric | N=20 | N=50 |
|---|---:|---:|
| Mean error (um) | 112.336706 | 124.874099 |
| Median error (um) | 92.419294 | 102.119261 |
| p95 error (um) | 231.051949 | 306.938531 |
| Maximum error (um) | 330.749655 | 583.981970 |
| Exactly-three robust minima | 20 / 20 | 50 / 50 |
| Solver failures | 0 | 0 |

The N=50 values are read from
`validation_results/closed_loop_inverse_5000_n50/summary.json`. It completed
50 forward solves with zero ambiguous/rejected rows. Four predicted coordinates
were outside the generator's +/-500 um training range before FEM evaluation;
the largest absolute prediction was 569.571862 um.

## Comparison with N=1000

The earlier N=1000 experiment had a training MAE of about **119.15 um** and a
20-case closed-loop mean of about **163.39 um**. N=5000 improves the held-out
training MAE to **107.14 um** and the N=20 physical closed-loop mean to
**112.34 um**. The larger synthetic dataset therefore improved both the
tabular inverse metric and the forward-model loop-closure check.

## Caveats

- N=20 and N=50 are still limited closed-loop subsets, not exhaustive tests of
  the eight-dimensional displacement space.
- The MLP can predict slightly outside the sampled +/-500 um displacement
  range.
- `random_forest.joblib` is intentionally not tracked: it is large and was not
  the best model.
- The present inverse is trained on synthetic FEM data and still needs
  validation against independent Wolfram and experimental data.
