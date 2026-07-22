# Generated N=29995 dataset QA

**Conclusion: ML-ready for a first inverse-model experiment.**

This audit is read-only with respect to the generated dataset. It does not run FEM, generate samples, calibrate physics, or train a model.

## Integrity checks

Clean rows: **29995**; rejected rows: **0**; completely parsed clean rows: **29995**.

| Check | Result |
|---|---|
| Clean schema exact | PASS |
| Rejected schema exact | PASS |
| All clean rows parsed | PASS |
| No malformed/nonfinite values | PASS |
| No duplicate full rows/IDs/inputs | PASS |
| All displacement coordinates within bounds | PASS |
| Wolfram transform exact | PASS |
| All minima inside ±8 mm search square | PASS |
| All minima inside displaced vacuum domain | PASS |
| All clean separations ≥0.15 mm | PASS |
| Stored pairwise values recompute exactly | PASS |
| Polar-order violations are zero | PASS |

Exact duplicate minima-output triples: **0**.
Minimum electrode-surface clearance: **6.77952 mm**; minimum outer-boundary clearance: **43.4159 mm**.
Largest absolute minimum coordinate: **6.58257 mm** versus the 8 mm search half-width.
Maximum stored/recomputed pairwise-distance discrepancy: **0 m**.

## Requested distribution statistics

| Quantity | Minimum | Maximum | Mean | Standard deviation | Median |
|---|---:|---:|---:|---:|---:|
| Wolfram displacement coordinates (µm) | -499.993769 | 499.997626 | 0.223348903 | 288.865294 | 1.12566753 |
| Minimum x coordinates (mm) | -6.28143643 | 6.5825738 | -0.0012493279 | 2.67962933 | 0.003277795 |
| Minimum y coordinates (mm) | -6.32026098 | 6.37375096 | -0.00272851662 | 2.68417423 | -0.0166441262 |
| Minimum pairwise distance (mm) | 0.597306575 | 9.13235433 | 5.27077807 | 1.40457634 | 5.36580296 |

The full SI-unit table, including x/y and per-label summaries, is in `summary_stats.csv`.

## Duplicate and value diagnostics

- Duplicate complete rows: 0.
- Duplicate sample IDs: 0.
- Duplicate Wolfram displacement vectors: 0.
- Duplicate minima-output triples: 0.
- Malformed numeric cells: 0.
- NaN/inf cells or JSON values: 0.

## Distribution observations and cautions

- Aggregate Wolfram displacement standard deviation is 288.865 µm versus 288.675 µm for an ideal continuous uniform distribution.
- Aggregate minimum-coordinate mean is (-0.00124933, -0.00272852) mm; small nonzero finite-sample offsets are not a schema defect.
- The closest minimum pair is 0.597307 mm, 3.982 times the 0.15 mm rejection threshold.
- 17 rows are below 1 mm and 406 are below 2 mm. The closest is sample 24352 with 12 additional robust-rejected candidates; it remains clean under the documented threshold.
- 2231 clean rows contain at least one candidate rejected by robust quality rules; this is compatible with exactly three robust-accepted minima.
- No production row exercised the rejected split. The rejection path is unit-tested, but rare ambiguous cases remain possible in larger samples.
- Polar-angle labels are deterministic but have the usual 0/2π seam; a model near that seam can see a label permutation despite continuous physics.
- N=29995 is suitable for inverse-model experiments, not a final coverage claim for the full eight-dimensional displacement space.

## Plots

- `plots/displacement_coordinates.png`
- `plots/minima_positions.png`
- `plots/minima_coordinates.png`
- `plots/minimum_pairwise_distance.png`

## ML readiness

The dataset passes every file-integrity, numerical, geometry, separation, and deterministic-label check and is safe for a **first inverse-model experiment**.
This is not evidence that N=29995 fully covers the eight-dimensional input space or that the four-electrode FEM is a complete physical model of every reference branch. Keep row-5-like ambiguities quarantined if they appear in future generation.
