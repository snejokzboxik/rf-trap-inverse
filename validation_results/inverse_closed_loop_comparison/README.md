# v1 vs v2 inverse-model closed-loop FEM comparison

This fresh like-for-like check uses no new data, model fitting, calibration, or mesh sweep.

## Shared protocol

- Sample IDs (20): `522, 738, 741, 661, 412, 679, 627, 514, 860, 137, 812, 77, 637, 974, 939, 900, 281, 884, 762, 320`.
- Selection: saved test split IDs for mlp.
- Each model predicts raw Wolfram W1--W4 displacements; FEM receives `-[W3, W1, W4, W2]`.
- The forward check uses real-scale all-positive electrodes, a fixed grounded outer boundary, robust minima mode, and practical 500 µm central mesh.
- Recomputed and original minimum sets are compared by Hungarian assignment.
- v2 uses its persisted ±500 µm clipping; its raw pre-clipping excursions are reported separately.

## Results

| Model | Mean (µm) | Median (µm) | p95 (µm) | Max (µm) | Exactly three | Failures | Ambiguous/rejected | Raw outside range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_baseline_mlp | 163.393179 | 152.667583 | 293.842672 | 546.476744 | 20 / 20 | 0 | 0 | 4 |
| v2_tuned_clipped_mlp | 212.840690 | 168.331404 | 469.008815 | 512.424174 | 20 / 20 | 0 | 0 | 2 |

## Decision

**Recommendation: retain v1 baseline MLP.** Replacement requires clean equal topology, lower mean error, and no worse p95 or maximum error. This is deliberately stricter than held-out displacement MAE because this comparison measures physical loop closure.

The CSV files preserve every selected case, status, candidate count, and per-minimum error. The plots compare the same ranked cases and all matched minima.
