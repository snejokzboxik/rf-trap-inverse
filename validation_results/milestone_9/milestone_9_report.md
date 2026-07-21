# Milestone 9 targeted refinement and calibration report

## Scope and invariants

No ML or synthetic dataset generation was performed. Local meshing, geometry, voltage, electrode mapping, and output transforms are explicit named diagnostics. The default all-positive real-scale model was not overwritten.

## Best results

| stage | hypothesis | rows | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best raw geometry screen | `geometry__electrode-radius-12mm` | 3 | 3 | 3 | 0.812039 | 0.775725 | 1.49691 | 1.28389 | False |
| Best raw voltage screen | `voltage__basis-fit-milestone-9-refined` | 3 | 3 | 3 | 0.894693 | 1.04738 | 1.28616 | 1.25457 | False |
| Best combined rows 1--10 | `combined__baseline-real-scale__all-positive__perm1324__calibrated-output` | 10 | 10 | 10 | 1.07456 | 0.744172 | 5.53543 | 2.5312 | False |
| Best promoted | `combined__baseline-real-scale__all-positive__perm1324` | 20 | 20 | 20 | 1.07473 | 0.991542 | 5.97072 | 2.24664 | False |

## Best calibrated hypothesis

- Geometry: `real-scale-default`; center radius 21.48 mm, electrode radius 10 mm, outer radius 50 mm.
- Electrode mapping: FEM E1--E4 <- source 1,3,2,4.
- Voltage model: `all-positive`; electrodes (1.0, 1.0, 1.0, 1.0), outer 0 V.
- Diagnostic output transform: scale 1, rotation 0 deg, anisotropy ratio 1.
- Central mesh h: 500 um.

## Interpretation

The completed row-1 500-to-200 um comparison worsened mean error by 0.791%; the 200 um rows 2--3 attempts timed out. This is below the specified 10% material-improvement criterion.
The best rows 1--10 calibrated result has mean 1.07456 mm and max 5.53543 mm.
The corresponding untransformed real-scale baseline on rows 1--10 has mean 1.08238 mm and max 5.97072 mm.
The promoted rows 1--20 result has mean 1.07473 mm and max 5.97072 mm, with exactly-three topology in 20/20 rows.

The validation gate fails.

Because mesh, geometry, coordinate, and static-voltage calibration do not meet the gate, the leading missing assumption is the physical electrode/drive model: Data.txt is associated with an octupole, while this solver still represents four circular electrode holes with one scalar Dirichlet potential per electrode. The full rod count, RF phase/amplitude grouping, or three-dimensional/end-effect physics may be necessary. This is a diagnostic conclusion, not a silent model change.

Synthetic dataset generation is **not safe** unless the gate is passed and Data.txt agreement is independently confirmed.

## Runtime and promotion

Current resumed orchestration wall time: 15.121 s. Checkpointed solves from earlier passes were reused, and the interrupted-run wall time is not included. The best three hypotheses were promoted to rows 1--20; rows 1--50 were not used because repeating local-mesh direct solves for every calibrated case was not proportionate after the rows 1--10 gate failure.
