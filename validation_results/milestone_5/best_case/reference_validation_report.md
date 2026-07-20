# FEM-to-reference validation report

## Comparison convention

The FEM fixes electrode 1. Each source row is translated by `-d1`: solver inputs are the permuted `(di-d1)` coordinates and reference minima are compared as `minimum_absolute-d1`.
Electrode map: `FEM E1=source E1 -> FEM E2=source E3 -> FEM E3=source E2 -> FEM E4=source E4`. Polarity variant: `all-positive`. Pairing uses minimum-total-distance assignment.

All calculations use metres. Tables display errors in both micrometres
and millimetres. Failed rows are retained and excluded from error metrics.

## Summary

- Selected rows: `10`
- Completed rows: `10`
- Failed/incomplete rows: `0`
- Rows with exactly three pre-selection Hessian-valid minima: `5`
- Matched minima: `30`
- Mean error: `1086.87 µm / 1.08687 mm`
- Median error: `953.208 µm / 0.953208 mm`
- Maximum error: `5982.41 µm / 5.98241 mm`
- 95th-percentile error: `2204.48 µm / 2.20448 mm`
- Row statuses: `{'ok': 10}`
- Wall runtime: `18.303 s`

## Scale and boundary diagnostics

- FEM nominal centre radius: `21.48 mm`
- FEM electrode radius: `10 mm`
- FEM outer-boundary radius: `50 mm`
- FEM minima-search half-extent: `8 mm`
- FEM target mesh size: `2 mm`
- Electrode potentials E1--E4: `(1.0, 1.0, 1.0, 1.0) V`
- Reference minima outside the search square: `0` of `30`
- Reference minima outside the FEM outer circle: `0` of `30`
- Reference radial-distance median/range: `3.88548 mm` / `0.670894` to `5.34136 mm`
- Computed radial-distance range: `1.54614 to 5.50423 mm`
- Maximum electrode-1 translation applied to the selected rows: `0.543111 mm`

## Per-row errors

| row | status | FEM minima | Hessian-valid | exactly 3 physical | mean error (µm / mm) | median error (µm / mm) | max error (µm / mm) | failure |
|---:|:---|---:|---:|:---:|---:|---:|---:|:---|
| 1 | ok | 3 | 3 | yes | 1248.47 µm / 1.24847 mm | 1254.59 µm / 1.25459 mm | 1479.49 µm / 1.47949 mm |  |
| 2 | ok | 3 | 4 | no | 843.555 µm / 0.843555 mm | 881.659 µm / 0.881659 mm | 905.327 µm / 0.905327 mm |  |
| 3 | ok | 3 | 3 | yes | 705.137 µm / 0.705137 mm | 976.4 µm / 0.9764 mm | 1010.32 µm / 1.01032 mm |  |
| 4 | ok | 3 | 5 | no | 759.58 µm / 0.75958 mm | 859.829 µm / 0.859829 mm | 1039.27 µm / 1.03927 mm |  |
| 5 | ok | 3 | 6 | no | 2485.66 µm / 2.48566 mm | 1019.81 µm / 1.01981 mm | 5982.41 µm / 5.98241 mm |  |
| 6 | ok | 3 | 3 | yes | 2060.36 µm / 2.06036 mm | 2183.77 µm / 2.18377 mm | 2221.42 µm / 2.22142 mm |  |
| 7 | ok | 3 | 3 | yes | 664.423 µm / 0.664423 mm | 936.253 µm / 0.936253 mm | 970.164 µm / 0.970164 mm |  |
| 8 | ok | 3 | 3 | yes | 265.356 µm / 0.265356 mm | 272.107 µm / 0.272107 mm | 360.387 µm / 0.360387 mm |  |
| 9 | ok | 3 | 4 | no | 427.118 µm / 0.427118 mm | 386.91 µm / 0.38691 mm | 702.58 µm / 0.70258 mm |  |
| 10 | ok | 3 | 5 | no | 1409.08 µm / 1.40908 mm | 1375.09 µm / 1.37509 mm | 1510.06 µm / 1.51006 mm |  |

## Per-minimum spatial assignment

| row | reference | computed | reference electrode-1-relative (mm) | computed electrode-1-relative (mm) | error (µm / mm) |
|---:|---:|---:|:---|:---|---:|
| 1 | 1 | 1 | (4.07535, 1.21358) | (2.67464, 1.68993) | 1479.49 µm / 1.47949 mm |
| 1 | 2 | 2 | (-4.34465, -0.026425) | (-3.35927, -0.25408) | 1011.33 µm / 1.01133 mm |
| 1 | 3 | 3 | (0.655351, 0.143575) | (1.25272, -0.959663) | 1254.59 µm / 1.25459 mm |
| 2 | 1 | 3 | (-0.021747, -2.86916) | (0.547997, -3.54201) | 881.659 µm / 0.881659 mm |
| 2 | 2 | 1 | (0.878253, 3.79084) | (0.664266, 2.91116) | 905.327 µm / 0.905327 mm |
| 2 | 3 | 2 | (-2.15175, 1.38084) | (-1.82662, 0.711995) | 743.677 µm / 0.743677 mm |
| 3 | 1 | 2 | (-2.58444, 3.23553) | (-1.99614, 2.41416) | 1010.32 µm / 1.01032 mm |
| 3 | 2 | 1 | (2.09556, 1.59553) | (2.81545, 2.25516) | 976.4 µm / 0.9764 mm |
| 3 | 3 | 3 | (-1.19444, -4.48447) | (-1.30847, -4.54411) | 128.687 µm / 0.128687 mm |
| 4 | 1 | 3 | (3.59298, -1.851) | (2.93629, -2.40604) | 859.829 µm / 0.859829 mm |
| 4 | 2 | 2 | (-4.05702, -1.001) | (-3.74985, -0.777912) | 379.637 µm / 0.379637 mm |
| 4 | 3 | 1 | (1.54298, 2.639) | (0.804394, 3.37016) | 1039.27 µm / 1.03927 mm |
| 5 | 1 | 2 | (-2.59047, -1.61851) | (-2.13575, -1.61226) | 454.762 µm / 0.454762 mm |
| 5 | 2 | 3 | (2.37953, -1.73851) | (2.99483, -2.55179) | 1019.81 µm / 1.01981 mm |
| 5 | 3 | 1 | (2.28953, -1.77851) | (-0.970163, 3.23782) | 5982.41 µm / 5.98241 mm |
| 6 | 1 | 2 | (-2.52921, -0.456952) | (-4.18317, 0.189718) | 1775.89 µm / 1.77589 mm |
| 6 | 2 | 1 | (1.29079, 5.18305) | (0.755357, 3.02712) | 2221.42 µm / 2.22142 mm |
| 6 | 3 | 3 | (1.27079, -2.71695) | (3.3316, -1.9945) | 2183.77 µm / 2.18377 mm |
| 7 | 1 | 2 | (-2.18353, -0.82716) | (-1.543, -0.0985114) | 970.164 µm / 0.970164 mm |
| 7 | 2 | 1 | (0.166465, 2.71284) | (-0.011585, 3.63201) | 936.253 µm / 0.936253 mm |
| 7 | 3 | 3 | (0.636465, -3.82716) | (0.611084, -3.7441) | 86.8535 µm / 0.0868535 mm |
| 8 | 1 | 1 | (-0.490301, 2.87009) | (-0.188, 2.67389) | 360.387 µm / 0.360387 mm |
| 8 | 2 | 2 | (-3.8003, -0.889909) | (-3.88549, -1.14834) | 272.107 µm / 0.272107 mm |
| 8 | 3 | 3 | (4.0297, -2.00991) | (4.05314, -1.84802) | 163.575 µm / 0.163575 mm |
| 9 | 1 | 2 | (-4.6232, 0.376757) | (-4.79574, 0.460672) | 191.865 µm / 0.191865 mm |
| 9 | 2 | 3 | (2.0468, -1.97324) | (2.74783, -2.01988) | 702.58 µm / 0.70258 mm |
| 9 | 3 | 1 | (3.3568, 2.17676) | (3.19941, 1.8233) | 386.91 µm / 0.38691 mm |
| 10 | 1 | 3 | (3.45753, -2.39561) | (2.35623, -3.21902) | 1375.09 µm / 1.37509 mm |
| 10 | 2 | 2 | (-4.81247, -1.61561) | (-3.47039, -1.61104) | 1342.08 µm / 1.34208 mm |
| 10 | 3 | 1 | (0.177535, 4.01439) | (-0.0715442, 5.50377) | 1510.06 µm / 1.51006 mm |

## Diagnostic interpretation

- **Electrode numbering:** the tested source-to-FEM permutation is
  stated above; the supplied data do not independently establish the map.
- **Coordinate origin:** the benchmark explicitly applies the selected
  absolute or electrode-1-relative convention rather than silently
  mixing frames.
- **Polarity convention:** the tested per-electrode potentials are
  reported above. The article's eight-rod octupole is not assumed to be
  equivalent to this four-electrode two-dimensional model.
- **Geometry scale:** all physical and numerical dimensions used by this
  run are reported above and must be judged against the measured errors.
- **Outer boundary and search region:** reference points outside the
  configured search square cannot be recovered by this solver run; points
  outside the finite outer circle are outside the modeled vacuum domain.

These results validate the current implementation against the supplied
data only at the stated conventions and reported geometry. They do not
authorize synthetic dataset generation unless the measured errors and
failure modes are resolved with the actual electrode geometry, polarity,
numbering, and boundary/search scales.
