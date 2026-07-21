# Generated N=1000 dataset QA

**Conclusion: ML-ready for a first inverse-model experiment.**

This audit is read-only with respect to the generated dataset. It does not run FEM, generate samples, calibrate physics, or train a model.

## Integrity checks

Clean rows: **1000**; rejected rows: **0**; completely parsed clean rows: **1000**.

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
Minimum electrode-surface clearance: **6.92508 mm**; minimum outer-boundary clearance: **43.7149 mm**.
Largest absolute minimum coordinate: **6.28344 mm** versus the 8 mm search half-width.
Maximum stored/recomputed pairwise-distance discrepancy: **0 m**.

## Requested distribution statistics

| Quantity | Minimum | Maximum | Mean | Standard deviation | Median |
|---|---:|---:|---:|---:|---:|
| Wolfram displacement coordinates (µm) | -499.978274 | 499.875733 | -6.2285747 | 284.865322 | -5.65385856 |
| Minimum x coordinates (mm) | -5.96835103 | 6.13308767 | 0.00565912007 | 2.69037325 | 0.0311969631 |
| Minimum y coordinates (mm) | -5.80179525 | 6.28343621 | -0.00113815856 | 2.68787253 | -0.049400555 |
| Minimum pairwise distance (mm) | 0.597349822 | 8.61880126 | 5.27542945 | 1.39499034 | 5.43603153 |

The full SI-unit table, including x/y and per-label summaries, is in `summary_stats.csv`.

## Duplicate and value diagnostics

- Duplicate complete rows: 0.
- Duplicate sample IDs: 0.
- Duplicate Wolfram displacement vectors: 0.
- Duplicate minima-output triples: 0.
- Malformed numeric cells: 0.
- NaN/inf cells or JSON values: 0.

## Distribution observations and cautions

- Aggregate Wolfram displacement standard deviation is 284.865 µm versus 288.675 µm for an ideal continuous uniform distribution.
- Aggregate minimum-coordinate mean is (0.00565912, -0.00113816) mm; small nonzero finite-sample offsets are not a schema defect.
- The closest minimum pair is 0.59735 mm, 3.982 times the 0.15 mm rejection threshold.
- 2 rows are below 1 mm and 13 are below 2 mm. The closest is sample 557 with 9 additional robust-rejected candidates; it remains clean under the documented threshold.
- 87 clean rows contain at least one candidate rejected by robust quality rules; this is compatible with exactly three robust-accepted minima.
- No production row exercised the rejected split. The rejection path is unit-tested, but rare ambiguous cases remain possible in larger samples.
- Polar-angle labels are deterministic but have the usual 0/2π seam; a model near that seam can see a label permutation despite continuous physics.
- N=1000 is suitable for a first inverse-model smoke experiment, not a final coverage claim for the full eight-dimensional displacement space.

## Plots

- `plots/displacement_coordinates.png`
- `plots/minima_positions.png`
- `plots/minima_coordinates.png`
- `plots/minimum_pairwise_distance.png`

## ML readiness

The dataset passes every file-integrity, numerical, geometry, separation, and deterministic-label check and is safe for a **first inverse-model experiment**.
This is not evidence that N=1000 fully covers the eight-dimensional input space or that the four-electrode FEM is a complete physical model of every reference branch. Keep row-5-like ambiguities quarantined if they appear in future generation.
