# Project status

## Already solved

- The real-scale four-electrode FEM forward model uses robust minima selection,
  a 500 um central mesh, all electrodes at +1 V, and the documented transform
  `F1,F2,F3,F4 = -[W3,W1,W4,W2]` from raw Wolfram displacements.
- The N=5000 and N=20000 synthetic datasets are complete and QA-passed. A
  provenance-preserving merged N=29995 dataset combines those rows with the
  non-overlapping `sample_id > 5000` tail of the seed-overlapping N=10000 run.
  The merged ML view has zero duplicate inputs, zero rejected rows, zero
  critical QA issues, and zero polar-order violations.
- The best saved inverse model at both scales is an MLP predicting raw
  Wolfram-order electrode displacements from three polar-angle-sorted minima.
- N=20000 closed-loop validation preserves exactly three robust minima in every
  checked N=50 and N=100 case, with zero solver failures and zero ambiguous
  branch rejections.
- The merged N=29995 MLP also preserves exactly three robust minima in all 100
  checked cases, with zero solver failures and zero ambiguous rejections.

## Current best metrics

| Measurement | Result |
|---|---:|
| Merged N=29995 MLP held-out MAE | 103.792467 um |
| Merged N=29995 MLP held-out RMSE | 132.485981 um |
| Merged N=29995 MLP maximum coordinate error | 542.911477 um |
| Merged N=29995 closed-loop mean, N=100 | 75.731039 um |
| Merged N=29995 closed-loop median, N=100 | 66.530430 um |
| Merged N=29995 closed-loop p95, N=100 | 163.043850 um |
| Merged N=29995 closed-loop maximum, N=100 | 373.952285 um |
| Merged N=29995 exactly-three topology, N=100 | 100 / 100 |

The merged model is the current preferred synthetic-data inverse and improves
the N=20000 N=100 closed-loop mean by about 21.4%. See
[MERGED_DATASET_29995_RESULTS.md](MERGED_DATASET_29995_RESULTS.md) for source
filters, duplicate checks, model comparisons, and caveats. The prior
[N20000_RESULTS.md](N20000_RESULTS.md) remains the unmerged benchmark.

## Next steps

1. Extend validation with larger deterministic held-out closed-loop subsets and
   retain every topology or solver failure explicitly.
2. Validate the synthetic-data inverse against independent Wolfram and, when
   available, experimental data before treating it as a physical calibration.
3. For any next dataset scale, repeat QA -> inverse training -> closed-loop
   N=50/N=100 -> comparison with N=20000 and merged N=29995 -> report update.

The project should continue to preserve the clean/rejected split, Wolfram-order
target convention, and physical closed-loop metric rather than relying only on
tabular inverse error.
