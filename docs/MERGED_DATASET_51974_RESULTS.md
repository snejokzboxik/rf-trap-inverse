# Merged N=51974 inverse reconstruction results

## Outcome

The provenance-preserving merge contains **51,974 clean samples** and is
ML-ready. The unchanged baseline training pipeline selected the MLP with a
held-out displacement-coordinate MAE of **102.891015 µm** and RMSE of
**132.217767 µm**. In the requested N=100 closed-loop FEM validation, all
100 predictions produced exactly three robust minima, with zero solver
failures and zero ambiguous-topology rejections. The matched-minima error was
**84.786605 µm mean**, **68.877023 µm median**, **212.973146 µm p95**, and
**536.915770 µm maximum**.

The larger dataset modestly improves held-out regression metrics relative to
N=29995, but it does **not** improve the observed N=100 closed-loop summary.
The two closed-loop runs used different deterministic subsets, so this is an
unpaired comparison and does not establish that adding data degraded the
physical reconstruction.

## Sources and convention

| Source | Input file | Included clean rows | Source seed(s) |
|---|---|---:|---|
| `generated_dataset_merged_29995` | `synthetic_clean_ml.csv` | 29,995 | 20260721, 20260723 |
| `generated_dataset_2000_probe` | `synthetic_clean.csv` | 2,000 | 20260725 |
| `generated_dataset_20000_semen` | `synthetic_clean.csv` | 19,979 | 31415 |
| **Total** |  | **51,974** |  |

Every source uses metres internally, maximum displacement **±500 µm**, and
the same canonical convention:

- Wolfram order: W1 upper-right, W2 lower-right, W3 upper-left, W4 lower-left.
- FEM order: F1 upper-left, F2 upper-right, F3 lower-left, F4 lower-right.
- Transform: `F1,F2,F3,F4 = -[W3,W1,W4,W2]`.

No source CSV was modified. `synthetic_clean.csv` adds `source_dataset`,
`source_sample_id`, `source_seed`, and contiguous `merged_sample_id` metadata.
`synthetic_clean_ml.csv` restores the exact schema required by inverse
training, using contiguous `sample_id` values 1 through 51,974.

## Source QA and Semen ID gaps

| Source | Clean | Rejected | Integrity result |
|---|---:|---:|---|
| Merged N=29995 | 29,995 | 0 | QA-passed |
| N=2000 probe | 2,000 | 0 | QA-passed |
| Semen N=20000 attempts | 19,979 | 21 | All clean-row integrity checks passed |

Semen's source-level QA originally reported `ml_ready=False` only because its
clean `sample_id` values are not contiguous after rejected attempts were
removed. The 21 absent IDs correspond exactly to 20 recorded solver failures
and one `not_exactly_three_robust_minima` rejection. They do not represent
missing or malformed clean rows. The 19,979 clean rows have exact schemas,
finite values, no duplicate inputs, in-range displacements, valid minima and
pairwise separation, the correct Wolfram transform, and zero polar-order
violations. Preserving those original IDs as `source_sample_id` while assigning
new contiguous merged IDs makes them safe for the merged training view without
altering the raw dataset.

The merged ML view passed its own QA:

| Check | Result |
|---|---:|
| Clean rows | 51,974 |
| Rejected rows in merged view | 0 |
| Critical issues | 0 |
| Polar-order violations | 0 |
| Duplicate complete rows | 0 |
| Duplicate Wolfram displacement inputs | 0 |
| Duplicate minima triples | 0 |
| Minimum pairwise separation | 0.305422 mm |
| ML-ready | Yes |

## Merge integrity

- `merged_sample_id` is exactly 1 through 51,974.
- Duplicate Wolfram displacement vectors: **0**.
- Duplicate `source_dataset + source_sample_id` pairs: **0**.
- All source summaries report `max_displacement_um=500`.
- All source summaries report `[-W3, -W1, -W4, -W2]`.
- Coordinate units are metres throughout.

## Inverse training

The existing deterministic 80/20 split used `random_state=42`, giving 41,579
training rows and 10,395 test rows. All three unchanged baseline models were
fit and evaluated.

| Model | MAE (µm) | RMSE (µm) | Maximum coordinate error (µm) | Fit time (s) |
|---|---:|---:|---:|---:|
| Ridge | 170.616817 | 209.663012 | 985.042242 | 0.028 |
| Random Forest | 129.513125 | 160.499889 | 638.749610 | 13.003 |
| **MLP** | **102.891015** | **132.217767** | **530.827205** | 111.476 |

The random forest was still included in the comparison, but its large joblib
artifact was intentionally not serialized. The small `mlp.joblib` and
`ridge.joblib` artifacts were retained.

## Closed-loop N=100

The closed-loop pipeline used the saved MLP, held-out test IDs, deterministic
selection state 20260725, batch size 5, the practical 500 µm central mesh, and
robust minima detection.

| Metric | N=51974 MLP |
|---|---:|
| Mean matched-minimum error | 84.786605 µm |
| Median | 68.877023 µm |
| p95 | 212.973146 µm |
| Maximum | 536.915770 µm |
| Exactly-three topology | 100 / 100 |
| Solver failures | 0 |
| Ambiguous/topology rejections | 0 |
| Predicted coordinates outside ±500 µm | 2 |
| Runtime | 205.497 s |

Here p95 is the error below which 95% of the 300 matched minimum errors fall.

## Comparison with merged N=29995

| Metric | N=29995 | N=51974 | Change |
|---|---:|---:|---:|
| Training/test MLP MAE (µm) | 103.792467 | **102.891015** | -0.901452 |
| Training/test MLP RMSE (µm) | 132.485981 | **132.217767** | -0.268214 |
| Training/test maximum (µm) | 542.911477 | **530.827205** | -12.084272 |
| Closed-loop N=100 mean (µm) | **75.731039** | 84.786605 | +9.055566 |
| Closed-loop N=100 median (µm) | **66.530430** | 68.877023 | +2.346594 |
| Closed-loop N=100 p95 (µm) | **163.043850** | 212.973146 | +49.929296 |
| Closed-loop N=100 maximum (µm) | **373.952285** | 536.915770 | +162.963485 |
| Exactly-three topology | 100 / 100 | 100 / 100 | unchanged |

The additional 21,979 independent clean rows helped the ordinary held-out
regression metrics slightly. They did not improve this independently selected
closed-loop N=100 summary, although topology remained fully stable. A fair
model-to-model physical comparison would run both saved MLPs on the same
sample IDs; that additional FEM work was intentionally outside this task.

## Artifacts

- Dataset: `validation_results/generated_dataset_merged_51974/`
- Dataset QA: `validation_results/generated_dataset_merged_51974_qa/`
- Inverse training: `validation_results/inverse_model_merged_51974/`
- Closed loop: `validation_results/closed_loop_inverse_merged_51974_n100/`

