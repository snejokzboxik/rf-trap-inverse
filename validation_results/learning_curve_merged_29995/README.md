# Merged N=29995 MLP learning curve

This ML-only experiment uses nested training subsets from one deterministic full-data training pool. Every point is evaluated on the same 5999-row held-out test set; no FEM solve, synthetic generation, or model artifact saving occurs.

| Requested N | Training rows | Fixed test rows | MAE (um) | RMSE (um) | Max (um) | Fit time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 800 | 5999 | 120.805389 | 150.065385 | 580.788191 | 7.351 |
| 5000 | 4000 | 5999 | 108.673243 | 138.338341 | 690.648183 | 58.611 |
| 10000 | 8000 | 5999 | 105.740198 | 134.646840 | 538.559993 | 26.486 |
| 20000 | 16000 | 5999 | 104.563974 | 133.533373 | 535.056322 | 46.968 |
| 29995 | 23996 | 5999 | 103.613591 | 132.388747 | 549.919969 | 103.571 |

MAE changes from 120.805389 um at the first point to 103.613591 um at N=29995, a 14.231% reduction.

`learning_curve_metrics.csv` contains the exact values. Plots are under `plots/`. The curve measures synthetic held-out regression only and does not replace independent or closed-loop physical validation.
