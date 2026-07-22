# Merged synthetic RF-trap dataset

This directory was created without modifying any source dataset.

## Sources and filters

- `generated_dataset_merged_29995`: 29995 rows; source sample IDs all..all; input `synthetic_clean_ml.csv`; seeds [20260721, 20260723].
- `generated_dataset_2000_probe`: 2000 rows; source sample IDs all..all; input `synthetic_clean.csv`; seeds [20260725].
- `generated_dataset_20000_semen`: 19979 rows; source sample IDs all..all; input `synthetic_clean.csv`; seeds [31415].

## Files

- `synthetic_clean.csv`: provenance-rich clean rows with merged and source IDs.
- `synthetic_clean_ml.csv`: exact original clean schema, with `sample_id` replaced by contiguous `merged_sample_id`; use this for QA and inverse training.
- `synthetic_rejected.csv`: empty exact-schema rejected view.
- `synthetic_summary.json`: source/filter and duplicate-check evidence.

Total rows: **51974**. Wolfram input duplicates: **0**. Source-pair duplicates: **0**.

The physical convention remains `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.
