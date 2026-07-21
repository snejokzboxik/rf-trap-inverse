# Wolfram displacement-convention check

The tested Wolfram convention treats `Data.txt` displacement pairs as W1=upper-right, W2=lower-right, W3=upper-left, W4=lower-left and applies:

`F1,F2,F3,F4 = -[raw_W3, raw_W1, raw_W4, raw_W2]`.

All four transformed vectors are added to the FEM nominal centers. The 50 mm grounded outer circle remains fixed at the origin. The run uses the real-scale all-positive geometry, robust minima, rows 1--10, and the existing 500 um local central mesh. No calibration or refinement sweep was run. Raw-absolute identity and perm1324 rows are reused from the previous focused check; only the Wolfram convention required new solves.

| convention | source order | sign | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `raw-absolute-identity` | 1-2-3-4 | 1 | 10/10 | 10/10 | 1.45386 | 1.12708 | 6.12674 | 2.38309 | False |
| `raw-absolute-perm1324` | 1-3-2-4 | 1 | 10/10 | 10/10 | 1.07921 | 0.879505 | 5.88481 | 2.32071 | False |
| `wolfram-signflip-reorder` | 3-1-4-2 | -1 | 10/10 | 10/10 | 0.250029 | 0.0444257 | 6.11783 | 0.121206 | False |

New Wolfram-convention wall time: 26.765 s.

Relative to raw absolute perm1324, the Wolfram convention changes mean error by +76.832%. This is a significant reduction under the 10% diagnostic threshold.

The validation gate does not pass.

The mean is 0.250029 mm, 0.000029 mm above the 0.25 mm limit. The maximum is concentrated in row 5: its median matched error is 0.0566488 mm but its maximum is 6.11783 mm, above the 0.5 mm limit.
