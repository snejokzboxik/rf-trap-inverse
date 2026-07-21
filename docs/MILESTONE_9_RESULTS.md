# Milestone 9: targeted local refinement and calibrated diagnostics

## Scope

Milestone 9 tests whether central mesh resolution or bounded calibration can make
the real-scale four-electrode FEM reproduce `Data.txt`. It does not add ML,
generate synthetic data, or replace the default physical model. Every geometry,
voltage, electrode mapping, and output transform remains a named diagnostic.

The production solver uses the robust-only Milestone 8 postprocessor in a fresh
process for each row. Evaluation checkpoints are atomic, so completed cases
survive interruption.

## Local central refinement

The Gmsh background field uses a 2 mm outer size, 0.5 mm electrode-boundary
size, and an 8 mm-radius central disk. This disk covers the rows 1--10 reference
minimum cloud. The default `real_scale_forward_config()` remains unchanged;
local refinement is opt-in through
`locally_refined_real_scale_forward_config()`.

| central h | completed rows | exactly three | mean reference error | evidence |
|---:|---:|---:|---:|---|
| 500 um | 3/3 | 3/3 | 0.913487 mm | reproduced after interruption |
| 200 um | 1/3 | 1/3 | 1.145611 mm (row 1) | rows 2--3 timed out in the interrupted run |
| 100 um | 0/3 | 0/3 | not available | interrupted concurrent attempt returned no completed rows |
| 50 um | 0/3 | 0/3 | not available | obsolete all-modes pilot exceeded 600 s and was terminated |
| 20 um and finer | 0/3 | 0/3 | not run | preflight and 50 um runtime evidence made direct solves impractical |

For row 1, refining from 500 to 200 um moves the largest matched branch by
30.1185 um and worsens mean reference error by 0.791%. This is not a three-row
convergence claim because rows 2--3 did not complete at 200 um. The 500 um
central mesh is selected for calibration because it is the only tested local
mesh with complete 3/3 robust topology evidence. Central resolution does not
meet the specified 10% improvement criterion.

The disk-only equilateral-element estimates are 1,858 triangles at 500 um,
11,609 at 200 um, 46,434 at 100 um, 185,734 at 50 um, 1,160,832 at 20 um, and
4.643 billion at 0.316228 um. These exclude transition, electrode, and outer
elements and are used only as preflight estimates.

## Geometry and voltage screens

Rows 1--3 are a screening set, not an independent validation set. Raw geometry
changes and output-coordinate calibration are reported separately.

- The best raw geometry-only screen uses a 12 mm electrode radius with the
  original 21.48 mm centre radius and 50 mm outer boundary: mean 0.812039 mm,
  maximum 1.496909 mm, exactly-three topology 3/3.
- The best raw voltage screen is the refined one-hot basis fit
  `(0.999795013, 1.0, 0.999977743, 0.999829473) V`: mean 0.894693 mm,
  maximum 1.286159 mm, exactly-three topology 3/3. It is again effectively the
  all-positive vector; voltage calibration does not identify a useful polarity
  correction.
- The checkerboard model is retained in the voltage table and is not promoted.
- A global voltage offset is removed only as a gauge shift applied to electrode
  and outer-boundary potentials together. Holding the outer boundary fixed while
  subtracting only the electrode mean would be a different physical model.

## Combined calibration

The combined search screens all six E1-preserving source permutations, curated
geometry ranges, normalized voltage variants, and bounded diagnostic output
transforms: global scale 0.7--1.3, rotation -15--15 degrees, and anisotropy ratio
0.85--1.15. The real-scale all-positive `E1,E3,E2,E4` baseline is always included.

| scope | hypothesis | completed / exactly-three | mean | median | maximum | p95 |
|---|---|---:|---:|---:|---:|---:|
| rows 1--10 | baseline, no output transform | 10/10 / 10/10 | 1.082384 mm | 0.937543 mm | 5.970718 mm | 2.330161 mm |
| rows 1--10 | baseline plus fitted output transform | 10/10 / 10/10 | **1.074555 mm** | 0.744172 mm | **5.535433 mm** | 2.531200 mm |
| rows 1--20 | baseline, no output transform | 20/20 / 20/20 | **1.074730 mm** | 0.991542 mm | 5.970718 mm | 2.246638 mm |

The rows 1--10 output transform is scale `0.91345324`, rotation
`-1.62011871 degrees`, and anisotropy ratio `1.13502527`. It reduces the
Milestone 5 mean by only about 1.13% and remains diagnostic. On rows 1--20 the
untransformed baseline ranks first, so the fitted transform does not generalize
as an improvement.

The best three completed rows 1--10 hypotheses were promoted only to rows
1--20. Promotion to rows 1--50 was not proportionate after the rows 1--10 gate
failure; checkpoints retain all completed cases for later extension if the
physical model changes.

## Validation decision

The gate requires mean error at most 0.25 mm, maximum error at most 0.5 mm, and
exactly three robust minima for every tested row. The best promoted case has
20/20 exactly-three topology but 1.074730 mm mean and 5.970718 mm maximum error.
The gate therefore fails decisively.

Local resolution, reasonable two-dimensional geometry calibration, electrode
numbering, static four-voltage combinations, and small global output transforms
do not reproduce `Data.txt`. Together with the Milestones 7--8 numerical audits,
the leading missing assumption is now the physical electrode/drive model. The
reference article describes an eight-rod octupole, whereas the current FEM has
four circular holes with one scalar Dirichlet value per electrode. Full rod
count and RF phase/amplitude grouping are the first missing assumptions to test;
three-dimensional/end effects are a secondary possibility.

`Data.txt` is not matched, so synthetic dataset generation remains unsafe.

