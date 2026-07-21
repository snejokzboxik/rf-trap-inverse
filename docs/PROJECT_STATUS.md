# Project status

## Already solved

- The real-scale four-electrode forward solver uses the documented Wolfram to
  FEM convention `-[W3,W1,W4,W2]`, robust minima selection, and a 500 um
  central mesh.
- A deterministic N=5000 synthetic dataset is complete, QA-passed, and has
  5000 clean rows with no rejected rows.
- The best saved N=5000 inverse model is an MLP trained to predict Wolfram-order
  electrode displacements from three polar-angle-sorted minima.
- Closed-loop validation preserves exactly three robust minima in all checked
  N=20 and N=50 cases, with zero solver failures.

## Current best metrics

| Measurement | Result |
|---|---:|
| N=5000 MLP held-out MAE | 107.140452 um |
| N=5000 MLP held-out RMSE | 137.553139 um |
| N=5000 closed-loop mean, N=20 | 112.336706 um |
| N=5000 closed-loop mean, N=50 | 124.874099 um |
| N=5000 closed-loop p95, N=50 | 306.938531 um |
| N=5000 exactly-three topology, N=50 | 50 / 50 |

## After N=20000 finishes

1. Run dataset QA and keep ambiguous or rejected rows separate.
2. Train the inverse models using the QA-passed clean CSV.
3. Run closed-loop validation on N=50 and N=100 deterministic held-out rows.
4. Compare the N=20000 training and closed-loop metrics with N=5000.
5. Update the N=20000 results report, including caveats and any regression.

The next scale-up should preserve the current no-leakage clean/rejected split,
Wolfram-order target convention, and independent closed-loop metric rather than
relying only on tabular inverse error.
