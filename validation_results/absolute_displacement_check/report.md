# Absolute four-electrode displacement check

`Data.txt` rows are applied in the fixed outer-boundary frame: each electrode center equals its nominal center plus its raw displacement. Electrode 1 moves; the grounded outer circle remains centered at the origin.

The check uses the real-scale all-positive geometry, 500 um local central mesh, robust minima mode, rows 1--10, and no geometry, voltage, or output calibration.

| mapping | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | change vs prior 1.07456 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `absolute-identity` | 10/10 | 10/10 | 1.45386 | 1.12708 | 6.12674 | 2.38309 | -35.299% |
| `absolute-perm1324` | 10/10 | 10/10 | 1.07921 | 0.879505 | 5.88481 | 2.32071 | -0.433% |

Focused wall time: 42.199 s.

The best absolute-displacement result does not reduce the prior approximately 1.07 mm rows 1--10 mismatch. This is a convention correction only; the validation gate and physical-model interpretation must be judged from the metrics above.
