# Merged synthetic RF-trap dataset

This directory was created without modifying any source dataset.

## Sources and filters

- `generated_dataset_5000`: 5000 rows; source sample IDs 1..all; seed 20260721.
- `generated_dataset_20000`: 20000 rows; source sample IDs 1..all; seed 20260723.
- `generated_dataset_10000`: 4995 rows; source sample IDs 5001..all; seed 20260721.

## Files

- `synthetic_clean.csv`: provenance-rich clean rows with merged and source IDs.
- `synthetic_clean_ml.csv`: exact original clean schema, with `sample_id` replaced by contiguous `merged_sample_id`; use this for QA and inverse training.
- `synthetic_rejected.csv`: empty exact-schema rejected view.
- `synthetic_summary.json`: source/filter and duplicate-check evidence.

Total rows: **29995**. Wolfram input duplicates: **0**. Source-pair duplicates: **0**.

The physical convention remains `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.
