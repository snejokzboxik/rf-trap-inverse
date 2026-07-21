# Milestone 9 local central-refinement report

The outer domain remains coarse at 2 mm, electrode boundaries use 0.5 mm, and only the 8 mm-radius central disk is refined. The FEM physics and robust-minima rules are unchanged.

This report resumes an interrupted run. Completed configurations were reproduced with the robust-only worker so row-level coordinates could be serialized; prior timeout/failure/termination records were preserved and were not counted as successful convergence evidence.

| central h (um) | completed | exactly three | mean reference error (mm) | max branch shift (um) | runtime (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 3/3 | 3/3 | 0.913487 | nan | 22.557 |
| 200 | 1/3 | 1/3 | 1.14561 | 30.1185 | 22.542 |
| 100 | 0/3 | 0/3 | nan | nan | 0.000 |
| 50 | 0/3 | 0/3 | nan | nan | 0.000 |
| 20 | 0/3 | 0/3 | nan | nan | 0.000 |
| 10 | 0/3 | 0/3 | nan | nan | 0.000 |
| 5 | 0/3 | 0/3 | nan | nan | 0.000 |
| 1 | 0/3 | 0/3 | nan | nan | 0.000 |
| 0.316228 | 0/3 | 0/3 | nan | nan | 0.000 |

Chosen practical central h: **500 um**. It is the only tested local mesh with completed exactly-three evidence for all rows 1--3; the 200 um result is row 1 only.
Row-1 mean-error change from 500 to 200 um: **0.791% worsening**. This is not a three-row convergence estimate because rows 2--3 timed out.

Meshes below the direct-solve preflight limit were run. Finer requests were preserved as skipped rows, not silently omitted.
- 200 um: interrupted-session robust-only worker did not return within the then-current 300 s cap; no mesh counts or coordinates were retained.
- 100 um: interrupted concurrent robust-only attempt completed 0/3 rows; the process-level failure detail was not persisted.
- 50 um: obsolete all-modes pilot exceeded 600 s and was terminated; it returned no mesh counts, minima, or error metrics.
- 20 um: preflight estimate 1,160,832 central triangles plus the 50 um runtime evidence made this direct solve impractical.
- 10 um: preflight estimate 4,643,327 central triangles plus the 50 um runtime evidence made this direct solve impractical.
- 5 um: preflight estimate 18,573,306 central triangles plus the 50 um runtime evidence made this direct solve impractical.
- 1 um: preflight estimate 464,332,638 central triangles plus the 50 um runtime evidence made this direct solve impractical.
- 0.316228 um: preflight estimate 4,643,326,373 central triangles plus the 50 um runtime evidence made this direct solve impractical.

A 10% reduction with complete three-row topology was the promotion criterion for resolution alone. That criterion was not met; central resolution alone is insufficient.
