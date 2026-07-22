# Merged N=29995 error-analysis outputs

This directory is derived only from saved ML predictions, saved closed-loop results, and the existing merged dataset. It runs no FEM solve and changes no numerical result.

- MLP held-out rows: 5999.
- Held-out coordinate MAE/RMSE/max: 103.792467 / 132.485981 / 542.911477 um.
- Hardest coordinate by MAE: w1_dy (105.433383 um).
- Hardest electrode by mean vector error: W1 (165.089613 um).
- Closed-loop matched-minimum mean/median/p95/max: 75.731039 / 66.530430 / 163.043850 / 373.952285 um.

Tabular outputs:

- `per_coordinate_error_stats.csv`
- `per_electrode_error_stats.csv`
- `closed_loop_error_stats.csv`
- `closed_loop_case_metrics.csv`
- `relationship_stats.csv`
- `worst_10_closed_loop_cases.csv`
- `analysis_summary.json`

Plots are under `plots/`. Correlations describe this deterministic N=100 closed-loop subset and do not establish physical causality.
