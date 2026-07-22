# Semen-ready FEM-order prediction dataset

This file is FEM-order only and is ready for direct FEM substitution.
Do not apply the Wolfram-to-FEM transform again.

- Rows: 100 (the first 100 rows of the source export; no randomization).
- Columns: 22 numeric columns only.
- All values are in metres.
- First 8 columns: true FEM displacements in F1, F2, F3, F4 order.
- Middle 6 columns: the three equilibrium minima, copied unchanged in canonical atan2 order.
- Last 8 columns: predicted FEM displacements in F1, F2, F3, F4 order.

Both displacement blocks already use:

`F1 = -W3, F2 = -W1, F3 = -W4, F4 = -W2`

Source: `validation_results/prediction_export_merged_51974/prediction_dataset_300.csv`
