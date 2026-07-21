# Synthetic RF-trap forward dataset

This directory contains 1000 deterministic forward samples drawn
with NumPy's `default_rng` seed `123`. Each raw displacement
coordinate is uniform on `[-500,
+500]` µm and is stored in metres.

## Coordinate convention

Raw inputs use Wolfram order: W1 upper-right, W2 lower-right, W3 upper-left,
and W4 lower-left. The absolute FEM displacement order is
`F1,F2,F3,F4 = -[W3,W1,W4,W2]`. All four electrodes move; the 50 mm grounded
outer circle remains fixed at the geometric origin. The electrodes are 10 mm
radius, their centre radius is 21.48 mm, all electrodes are +1 V, and the outer
boundary is 0 V.

Outputs are robust pseudopotential minima in absolute geometric-centre
coordinates, sorted by polar angle `atan2(y,x)` mapped to `[0,2*pi)`. The
practical mesh is a 500 µm central refinement with a coarse outer domain.

## Split policy

`synthetic_clean.csv` contains only samples with exactly three robust-accepted
candidates and minimum pairwise separation at least 0.15 mm.
`synthetic_rejected.csv` preserves solver failures, invalid geometry,
non-three robust topology, and `ambiguous_branch` cases. No rejected sample is
silently included in the clean file. `rejected_candidate_count` instead counts
individual candidates discarded by robust quality rules; a clean solve may have
such candidates provided exactly three candidates were robust-accepted.
`Data.txt` row 5 is not used as a training row; this dataset is sampled
independently.

Counts: clean `1000`, rejected
`0`, ambiguous branch
`0`. Status counts: `{'clean': 1000}`.

## Files

- `synthetic_clean.csv`: stable 27-column training table.
- `synthetic_rejected.csv`: the same core fields plus failure/topology audit data.
- `synthetic_summary.json`: configuration, counts, convention, and runtimes.
