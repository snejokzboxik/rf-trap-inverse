# Merged N=29995 inverse reconstruction results

This report records the first provenance-preserving merge of the existing
synthetic RF-trap datasets and the inverse/closed-loop validation performed on
the merged ML view. No source dataset was overwritten and no new FEM samples
were generated for the merge.

## Sources and merge rules

| Source dataset | Seed | Filter | Included clean rows |
|---|---:|---|---:|
| `generated_dataset_5000` | 20260721 | all clean rows | 5000 |
| `generated_dataset_20000` | 20260723 | all clean rows | 20000 |
| `generated_dataset_10000` | 20260721 | `sample_id > 5000` | 4995 |
| **Merged total** |  |  | **29995** |

The N=10000 source shares seed 20260721 with the N=5000 dataset and overlaps
it for 4998 sample IDs. Only IDs above 5000 are therefore included as new
samples. Five of the nominal IDs above 5000 are absent from the clean file
because their original FEM solves failed, leaving 4995 usable rows.

The provenance-rich `synthetic_clean.csv` preserves every original ID as
`source_sample_id` and adds `source_dataset`, `source_seed`, and a contiguous
`merged_sample_id`. The exact-schema `synthetic_clean_ml.csv` replaces
`sample_id` with that contiguous merged ID and omits provenance metadata so it
can be consumed by the existing QA and inverse-training tools.

The physical convention is unchanged: raw inputs are in Wolfram electrode
order and the absolute-displacement transform into FEM order is
`F1,F2,F3,F4 = -[W3,W1,W4,W2]`.

## Integrity and QA

| Check | Result |
|---|---:|
| Merged clean rows | 29995 |
| `merged_sample_id` range | 1..29995 |
| Duplicate eight-coordinate Wolfram inputs | 0 |
| Duplicate `source_dataset + source_sample_id` pairs | 0 |
| QA `ml_ready` | True |
| QA rejected rows | 0 |
| QA critical issues | 0 |
| QA polar-order violations | 0 |

The merged files are under
`validation_results/generated_dataset_merged_29995`; the read-only QA outputs
are under `validation_results/generated_dataset_merged_29995_qa`.

## Inverse training

The standard deterministic split (`test_size=0.2`, `random_state=42`) contains
23996 training rows and 5999 test rows. MLP again ranks first by held-out MAE.
All errors are in micrometres.

| Dataset/model | Test MAE | Test RMSE | Max absolute coordinate error |
|---|---:|---:|---:|
| N=5000 MLP | 107.140452 | 137.553139 | 508.514970 |
| N=20000 MLP | 104.823850 | 134.553068 | 519.119544 |
| **Merged N=29995 MLP** | **103.792467** | **132.485981** | **542.911477** |

The merged MLP improves MAE and RMSE relative to N=20000, but its worst
single-coordinate error is higher. The MLP and ridge artifacts are retained;
the roughly 1.17 GB random-forest artifact is intentionally omitted because
random forest was not the best model.

## Closed-loop FEM validation

The closed loop uses true minima -> predicted Wolfram displacements ->
`-[W3,W1,W4,W2]` -> robust forward FEM -> Hungarian matching to the original
minima. The merged model uses a deterministic N=100 subset with random state
20260723 and the practical 500 um central mesh.

| Metric (um) | N=5000, N=50 | N=20000, N=100 | Merged N=29995, N=100 |
|---|---:|---:|---:|
| Mean | 124.874099 | 96.296420 | **75.731039** |
| Median | 102.119261 | 79.384966 | **66.530430** |
| p95 | 306.938531 | 212.087844 | **163.043850** |
| Maximum | 583.981970 | 403.643453 | **373.952285** |
| Exactly-three topology | 50 / 50 | 100 / 100 | **100 / 100** |
| Solver failures | 0 | 0 | **0** |
| Ambiguous/rejected rows | 0 | 0 | **0** |

The merged N=100 closed loop improves the N=20000 N=100 mean error by about
21.4%, while preserving valid three-minima topology for every selected row.
Three predicted displacement coordinates exceeded the +/-500 um training
range before FEM evaluation; the largest absolute prediction was 517.41 um.

## Conclusion

The provenance-preserving merge is internally consistent, ML-ready, and free
of duplicate displacement inputs. The merged MLP is the current preferred
synthetic-data inverse: it has the best held-out MAE and the best physical
closed-loop metrics measured so far, including a 75.73 um mean minimum-position
error on 100 samples. The maximum tabular coordinate error did not improve,
and the validation is still against the same synthetic FEM model. Independent
Wolfram and experimental validation remains necessary before treating the
inverse as a physical calibration.
