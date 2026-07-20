# Milestone 8 robust-minima validation report

## Decision

Robust post-processing removes the tested selected-minimum numerical confounder,
while the reference gate still fails. The remaining mismatch can therefore be
attributed to model class/topology with substantially stronger numerical justification.
Three coarse 2.0-to-1.5 mm branch shifts exceed the reporting tolerance, so this
is stronger evidence rather than a claim of exact mesh invariance.

The validation gate remains failed.
No physical FEM model, voltage convention, or default recovered-gradient behavior was changed.

## Mode and mesh comparison

| mode | mapping | h (mm) | completed | exactly three | mean (mm) | median (mm) | max (mm) | p95 (mm) | selected flags | rejected | gate |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| audit-flag-identity | identity | 2 | 10/10 | 5/10 | 1.47368 | 1.18772 | 6.20446 | 2.44766 | 5 | 0 | fail |
| audit-flag-perm1324 | perm1324 | 2 | 10/10 | 5/10 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 4 | 0 | fail |
| old-recovered-identity | identity | 2 | 10/10 | 5/10 | 1.47368 | 1.18772 | 6.20446 | 2.44766 | 0 | 0 | fail |
| old-recovered-perm1324-comparison | perm1324 | 2 | 10/10 | 5/10 | 1.08687 | 0.953208 | 5.98241 | 2.20448 | 0 | 0 | fail |
| robust-identity | identity | 2 | 10/10 | 10/10 | 1.47427 | 1.1858 | 6.20776 | 2.44758 | 0 | 26 | fail |
| robust-perm1324 | perm1324 | 0.5 | 3/3 | 3/3 | 0.921771 | 1.00037 | 1.29588 | 1.22524 | 0 | 0 | fail |
| robust-perm1324 | perm1324 | 0.75 | 10/10 | 10/10 | 1.08506 | 0.925806 | 5.92962 | 2.32698 | 0 | 8 | fail |
| robust-perm1324 | perm1324 | 1 | 10/10 | 10/10 | 1.073 | 0.924126 | 5.90851 | 2.29576 | 0 | 7 | fail |
| robust-perm1324 | perm1324 | 1.5 | 10/10 | 10/10 | 1.09009 | 0.928404 | 5.92913 | 2.29613 | 0 | 10 | fail |
| robust-perm1324 | perm1324 | 2 | 10/10 | 10/10 | 1.08754 | 0.956273 | 5.98525 | 2.20475 | 0 | 24 | fail |

Audit-flag mode deliberately reuses the old coordinates and only exposes the
Milestone 7 conservative flags (5 identity flags;
4 best-mapping flags, reproducing rows 3, 4, 5, and 10).
Robust identity changes the mean error by 0.039% relative to old identity;
robust best-mapping changes it by 0.061% relative to the same-mapping old result.
The changes are classified as not material at the documented 5% threshold.

## Robust criteria

Candidate sources are the legacy recovered coarse/refined search, exact zeros of the
continuous recovered P1 vector field inside mesh triangles, and local lows of the raw
elementwise-constant P1 field. Sources within the configured merge distance are combined
with provenance retained.

Robust acceptance requires a positive, stable Hessian at all valid mesh-scaled stencils
and recovered |E|^2 no more than 100 times the low-candidate scale. A candidate close to
an internal facet is rejected as facet-sensitive only when the adjacent raw-field jump is
large and the multi-stencil Hessian is unstable. All rejected candidates remain in the CSV.
The artifact probability is an uncalibrated rule score (facet 0.25, raw jump 0.20,
Hessian instability 0.40, high recovered |E|^2 0.15, capped at one); no reference
coordinate enters the score or robust acceptance decision.

## Topology and mesh consistency

Exactly-three topology under the best mapping changes from 5/10
to 10/10; this is an improvement.
Successive-mesh branch matches stable within 0.25 mm: 96/99.
Maximum successive branch shift: 0.2998 mm.
For transitions at h <= 1.5 mm, stability is 69/69
with maximum shift 0.164062 mm.

## Scientific interpretation

Selected recovered-gradient artifacts are not responsible for a significant fraction of the spatial mismatch.
Robust selection can change candidate topology without forcing agreement with Data.txt.
The conservative gate requires all rows complete, exactly-three topology for every row,
mean error <= 0.25 mm, and maximum error <= 0.5 mm.

Synthetic dataset generation remains unsafe because the validation gate is not met.

## Reproducibility

Runtime: 300.510 seconds. Every FEM case ran in a fresh process.
CSV coordinates and calculations use SI units; millimetre columns are presentation copies.
