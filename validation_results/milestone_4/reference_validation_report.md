# FEM-to-reference validation report

## Comparison convention

The FEM fixes electrode 1. Each source row is therefore translated by
`-d1`: solver inputs are `(d2-d1, d3-d1, d4-d1)` and reference minima
are compared as `minimum_absolute-d1`. The source ordering is retained
for identifiers, but pairing uses minimum-total-distance assignment.

All calculations use metres. Tables display errors in both micrometres
and millimetres. Failed rows are retained and excluded from error metrics.

## Summary

- Selected rows: `10`
- Completed rows: `8`
- Failed/incomplete rows: `2`
- Rows with exactly three pre-selection Hessian-valid minima: `7`
- Matched minima: `24`
- Mean error: `3185.85 µm / 3.18585 mm`
- Median error: `3089.38 µm / 3.08938 mm`
- Maximum error: `4934.35 µm / 4.93435 mm`
- 95th-percentile error: `4559.83 µm / 4.55983 mm`
- Row statuses: `{'forward-failed': 2, 'ok': 8}`

## Scale and boundary diagnostics

- FEM nominal centre radius: `1.1 mm`
- FEM electrode radius: `0.32 mm`
- FEM outer-boundary radius: `4 mm`
- FEM minima-search half-extent: `0.7 mm`
- Reference minima outside the search square: `29` of `30`
- Reference minima outside the FEM outer circle: `13` of `30`
- Reference radial-distance median/range: `3.88548 mm` / `0.670894` to `5.34136 mm`
- Computed radial-distance range: `0.145615 to 0.84782 mm`
- Maximum electrode-1 translation applied to the selected rows: `0.543111 mm`

## Per-row errors

| row | status | FEM minima | Hessian-valid | exactly 3 physical | mean error (µm / mm) | median error (µm / mm) | max error (µm / mm) | failure |
|---:|:---|---:|---:|:---:|---:|---:|---:|:---|
| 1 | forward-failed | 0 | - | no | n/a | n/a | n/a | ValueError: electrode disks must not touch or overlap |
| 2 | ok | 3 | 3 | yes | 2731.82 µm / 2.73182 mm | 2674.73 µm / 2.67473 mm | 3372.77 µm / 3.37277 mm |  |
| 3 | forward-failed | 0 | - | no | n/a | n/a | n/a | MinimaSearchError: found 2 validated minima; expected 3 |
| 4 | ok | 3 | 3 | yes | 3313.65 µm / 3.31365 mm | 3257.71 µm / 3.25771 mm | 3975.65 µm / 3.97565 mm |  |
| 5 | ok | 3 | 3 | yes | 2659.01 µm / 2.65901 mm | 2652.49 µm / 2.65249 mm | 2921.05 µm / 2.92105 mm |  |
| 6 | ok | 3 | 3 | yes | 3092.49 µm / 3.09249 mm | 2448.86 µm / 2.44886 mm | 4582.69 µm / 4.58269 mm |  |
| 7 | ok | 3 | 3 | yes | 2568.94 µm / 2.56894 mm | 2354.87 µm / 2.35487 mm | 3733.27 µm / 3.73327 mm |  |
| 8 | ok | 3 | 3 | yes | 3546.67 µm / 3.54667 mm | 3832.9 µm / 3.8329 mm | 4231.99 µm / 4.23199 mm |  |
| 9 | ok | 3 | 5 | no | 3489.26 µm / 3.48926 mm | 3521.31 µm / 3.52131 mm | 4934.35 µm / 4.93435 mm |  |
| 10 | ok | 3 | 3 | yes | 4084.98 µm / 4.08498 mm | 3945.66 µm / 3.94566 mm | 4430.33 µm / 4.43033 mm |  |

## Per-minimum spatial assignment

| row | reference | computed | reference relative (mm) | computed relative (mm) | error (µm / mm) |
|---:|---:|---:|:---|:---|---:|
| 2 | 1 | 3 | (-0.021747, -2.86916) | (0.0645491, -0.195824) | 2674.73 µm / 2.67473 mm |
| 2 | 2 | 1 | (0.878253, 3.79084) | (0.38202, 0.454773) | 3372.77 µm / 3.37277 mm |
| 2 | 3 | 2 | (-2.15175, 1.38084) | (-0.585926, -0.0895165) | 2147.96 µm / 2.14796 mm |
| 4 | 1 | 3 | (3.59298, -1.851) | (0.590436, -0.587102) | 3257.71 µm / 3.25771 mm |
| 4 | 2 | 2 | (-4.05702, -1.001) | (-0.362551, 0.467568) | 3975.65 µm / 3.97565 mm |
| 4 | 3 | 1 | (1.54298, 2.639) | (0.238061, 0.266629) | 2707.58 µm / 2.70758 mm |
| 5 | 1 | 2 | (-2.59047, -1.61851) | (-0.392552, -0.645947) | 2403.48 µm / 2.40348 mm |
| 5 | 2 | 1 | (2.37953, -1.73851) | (0.300639, 0.313499) | 2921.05 µm / 2.92105 mm |
| 5 | 3 | 3 | (2.28953, -1.77851) | (-0.0687885, -0.564408) | 2652.49 µm / 2.65249 mm |
| 6 | 1 | 2 | (-2.52921, -0.456952) | (-0.363696, 0.138623) | 2245.92 µm / 2.24592 mm |
| 6 | 2 | 1 | (1.29079, 5.18305) | (0.340357, 0.7) | 4582.69 µm / 4.58269 mm |
| 6 | 3 | 3 | (1.27079, -2.71695) | (0.7, -0.335544) | 2448.86 µm / 2.44886 mm |
| 7 | 1 | 2 | (-2.18353, -0.82716) | (-0.625718, -0.387456) | 1618.68 µm / 1.61868 mm |
| 7 | 2 | 1 | (0.166465, 2.71284) | (0.337931, 0.364225) | 2354.87 µm / 2.35487 mm |
| 7 | 3 | 3 | (0.636465, -3.82716) | (-0.305125, -0.214585) | 3733.27 µm / 3.73327 mm |
| 8 | 1 | 1 | (-0.490301, 2.87009) | (-0.326632, 0.300182) | 2575.12 µm / 2.57512 mm |
| 8 | 2 | 2 | (-3.8003, -0.889909) | (0.0278958, -0.7) | 3832.9 µm / 3.8329 mm |
| 8 | 3 | 3 | (4.0297, -2.00991) | (0.157898, -0.301436) | 4231.99 µm / 4.23199 mm |
| 9 | 1 | 2 | (-4.6232, 0.376757) | (0.300552, 0.7) | 4934.35 µm / 4.93435 mm |
| 9 | 2 | 3 | (2.0468, -1.97324) | (0.7, -0.478328) | 2012.12 µm / 2.01212 mm |
| 9 | 3 | 1 | (3.3568, 2.17676) | (0.43585, 0.210113) | 3521.31 µm / 3.52131 mm |
| 10 | 1 | 3 | (3.45753, -2.39561) | (0.00347769, -0.630431) | 3878.97 µm / 3.87897 mm |
| 10 | 2 | 2 | (-4.81247, -1.61561) | (-0.699383, 0.0307026) | 4430.33 µm / 4.43033 mm |
| 10 | 3 | 1 | (0.177535, 4.01439) | (0.128207, 0.0690405) | 3945.66 µm / 3.94566 mm |

## Diagnostic interpretation

- **Electrode numbering:** source numbering is assumed to match FEM
  numbering; the supplied data do not independently establish this map.
- **Coordinate origin / absolute versus electrode-1-relative:** the
  benchmark explicitly applies the required electrode-1 translation.
  Its magnitude is reported above, so remaining multi-millimetre scale
  disagreement cannot be attributed solely to this convention.
- **Polarity convention:** the current FEM drives four electrodes at
  equal phase, whereas the reference article describes an eight-rod
  alternating-polarity octupole. Equivalence is not established.
- **Geometry scale:** the FEM radius and nominal centres are explicitly
  provisional. Reference minima several millimetres from the origin are
  incompatible with treating the current demonstrator dimensions as
  validated physical geometry.
- **Outer boundary and search region:** reference points outside the
  configured search square cannot be recovered by this solver run; points
  outside the finite outer circle are outside the modeled vacuum domain.

These results validate the current implementation against the supplied
data only at the stated conventions and provisional geometry. They do not
authorize synthetic dataset generation unless the measured errors and
failure modes are resolved with the actual electrode geometry, polarity,
numbering, and boundary/search scales.
