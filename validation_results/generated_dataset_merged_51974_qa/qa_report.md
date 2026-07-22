# Generated N=51974 dataset QA

**Conclusion: ML-ready for a first inverse-model experiment.**

This audit is read-only with respect to the generated dataset. It does not run FEM, generate samples, calibrate physics, or train a model.

## Integrity checks

Clean rows: **51974**; rejected rows: **0**; completely parsed clean rows: **51974**.

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
| Wolfram displacement coordinates (µm) | -499.994191 | 499.997626 | 0.264318642 | 288.702467 | 1.04103216 |
| Minimum x coordinates (mm) | -6.28979311 | 6.5825738 | -0.00080621089 | 2.68496082 | 0.00104509443 |
| Minimum y coordinates (mm) | -6.36624452 | 6.37375096 | -0.00350449513 | 2.67850379 | -0.0100729448 |
| Minimum pairwise distance (mm) | 0.305421686 | 9.13235433 | 5.27297715 | 1.40154657 | 5.37252695 |

The full SI-unit table, including x/y and per-label summaries, is in `summary_stats.csv`.

## Duplicate and value diagnostics

- Duplicate complete rows: 0.
- Duplicate sample IDs: 0.
- Duplicate Wolfram displacement vectors: 0.
- Duplicate minima-output triples: 0.
- Malformed numeric cells: 0.
- NaN/inf cells or JSON values: 0.

## Distribution observations and cautions

- Aggregate Wolfram displacement standard deviation is 288.702 µm versus 288.675 µm for an ideal continuous uniform distribution.
- Aggregate minimum-coordinate mean is (-0.000806211, -0.0035045) mm; small nonzero finite-sample offsets are not a schema defect.
- The closest minimum pair is 0.305422 mm, 2.036 times the 0.15 mm rejection threshold.
- 35 rows are below 1 mm and 714 are below 2 mm. The closest is sample 45075 with 8 additional robust-rejected candidates; it remains clean under the documented threshold.
- 3943 clean rows contain at least one candidate rejected by robust quality rules; this is compatible with exactly three robust-accepted minima.
- No production row exercised the rejected split. The rejection path is unit-tested, but rare ambiguous cases remain possible in larger samples.
- Polar-angle labels are deterministic but have the usual 0/2π seam; a model near that seam can see a label permutation despite continuous physics.
- N=51974 is suitable for inverse-model experiments, not a final coverage claim for the full eight-dimensional displacement space.

## Plots

- `plots/displacement_coordinates.png`
- `plots/minima_positions.png`
- `plots/minima_coordinates.png`
- `plots/minimum_pairwise_distance.png`

## ML readiness

The dataset passes every file-integrity, numerical, geometry, separation, and deterministic-label check and is safe for a **first inverse-model experiment**.
This is not evidence that N=51974 fully covers the eight-dimensional input space or that the four-electrode FEM is a complete physical model of every reference branch. Keep row-5-like ambiguities quarantined if they appear in future generation.
