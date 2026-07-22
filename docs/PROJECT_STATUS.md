# Project status

## Already solved

- The real-scale four-electrode FEM forward model uses robust minima selection,
  a 500 um central mesh, all electrodes at +1 V, and the documented transform
  `F1,F2,F3,F4 = -[W3,W1,W4,W2]` from raw Wolfram displacements.
- The N=5000 and N=20000 synthetic datasets are complete and QA-passed. Each
  has only clean rows (5000 and 20000 respectively), zero rejected rows, zero
  critical QA issues, and zero polar-order violations.
- The best saved inverse model at both scales is an MLP predicting raw
  Wolfram-order electrode displacements from three polar-angle-sorted minima.
- N=20000 closed-loop validation preserves exactly three robust minima in every
  checked N=50 and N=100 case, with zero solver failures and zero ambiguous
  branch rejections.

## Current best metrics

| Measurement | Result |
|---|---:|
| N=20000 MLP held-out MAE | 104.823850 um |
| N=20000 MLP held-out RMSE | 134.553068 um |
| N=20000 MLP maximum coordinate error | 519.119544 um |
| N=20000 closed-loop mean, N=50 | 93.430449 um |
| N=20000 closed-loop mean, N=100 | 96.296420 um |
| N=20000 closed-loop median, N=100 | 79.384966 um |
| N=20000 closed-loop p95, N=100 | 212.087844 um |
| N=20000 exactly-three topology, N=100 | 100 / 100 |

N=20000 therefore provides stable closed-loop reconstruction below 100 um mean
minima-position error on the 100-sample check. See [N20000_RESULTS.md](N20000_RESULTS.md)
for the direct N=5000 comparison, assumptions, and caveats.

## Next steps

1. Extend validation with larger deterministic held-out closed-loop subsets and
   retain every topology or solver failure explicitly.
2. Validate the synthetic-data inverse against independent Wolfram and, when
   available, experimental data before treating it as a physical calibration.
3. For any next dataset scale, repeat QA -> inverse training -> closed-loop
   N=50/N=100 -> comparison with N=20000 -> results-report update.

The project should continue to preserve the clean/rejected split, Wolfram-order
target convention, and physical closed-loop metric rather than relying only on
tabular inverse error.
