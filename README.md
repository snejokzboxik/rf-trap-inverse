# RF trap forward model and reference benchmarking — milestone 8

This repository implements one reproducible 2D forward-model pipeline for four
infinitely long, equal-phase cylindrical electrodes.  Given the six Cartesian
displacements of electrodes 2–4, it:

1. builds a circular vacuum domain with four circular holes;
2. generates a conforming first-order triangular mesh with Gmsh;
3. solves Laplace's equation with scikit-fem;
4. recovers a continuous electric-field surrogate and evaluates `|E|²`;
5. coarse-scans, refines, merges, Hessian-validates, and polar-angle-sorts three
   local minima.

The recovered-gradient search remains the default for regression compatibility.
Milestone eight adds explicit `recovered-gradient`, `raw-element-diagnostic`, and
`robust` post-processing modes without changing the FEM solve or default physics.

The package also runs full mesh-size × outer-radius convergence studies, compares
successive minima with minimum-distance spatial assignment, and writes CSV,
Markdown, and PNG reports.

Milestone three adds deterministic ingestion of the supplied `Data.txt`
reference dataset. It preserves the eight raw electrode-displacement coordinates,
derives the six coordinates relative to electrode 1, keeps both absolute and
electrode-1-relative minima, and applies the forward solver's polar-angle sorting
convention.

Milestone four benchmarks selected source rows against isolated FEM solves. It
uses minimum-total-distance assignment, records failed configurations instead of
discarding them, and writes per-row/per-minimum errors plus comparison plots.

Milestone five adds the supplied real-scale geometry and controlled diagnostics
for mesh size, coordinate frame, electrode numbering, and per-electrode polarity.
The old demonstrator remains available explicitly for regression and convergence
tests, but reference validation now defaults to the real-scale configuration.

The package uses metres and volts internally. Geometry and numerical values are
configuration inputs. The dimensions in `examples/run_one_configuration.py`
remain the earlier demonstrator; reference validation uses the named real-scale
configuration described below.

## Real-scale geometry and electrode numbering

The real-scale configuration uses a 50 mm outer-boundary radius, 10 mm electrode
radius, and 11.48 mm from the trap centre to the nearest electrode surface. The
electrode-centre radius is therefore 21.48 mm. With
`a = 21.48 mm / sqrt(2) = 15.1886537 mm`, the fixed numbering convention is:

- E1 = `(-a, +a)` (upper left, reference electrode);
- E2 = `(+a, +a)` (upper right);
- E3 = `(-a, -a)` (lower left);
- E4 = `(+a, -a)` (lower right).

The six-component forward API always fixes E1 and accepts E2--E4 displacements.
Dataset validation can either translate raw coordinates into that E1-relative
frame or apply all four raw displacements in an explicit absolute-frame
diagnostic. Source-electrode permutations are named and reported; they never
silently change this FEM numbering.

## Install and run

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest
.venv\Scripts\python examples\run_one_configuration.py
.venv\Scripts\rf-trap-convergence
.venv\Scripts\rf-trap-investigate-extra-candidate
.venv\Scripts\rf-trap-reference-dataset
.venv\Scripts\rf-trap-reference-validation
.venv\Scripts\rf-trap-scale-validation
.venv\Scripts\rf-trap-hypothesis-validation
.venv\Scripts\rf-trap-fem-audit
.venv\Scripts\rf-trap-robust-minima-validation
```

The default convergence command evaluates mesh sizes of 120, 80, and 60 µm at
outer radii of 3.5, 4.0, and 5.0 mm. It writes:

- `convergence_runs.csv`, with mesh controls, counts, residuals, search
  diagnostics, coordinates, `|E|²`, and Hessian eigenvalues;
- `coordinate_comparisons.csv`, with assignment-matched successive shifts;
- `convergence_report.md`;
- `mesh_refinement_minima.png` and `outer_radius_minima.png`.

Use `rf-trap-convergence --help` to change the sweep, tolerance, or output
directory. `python -m rf_trap_forward.cli` is equivalent. Production convergence
cases run in fresh interpreter processes to prevent Gmsh run history from
confounding the comparison.

The one-configuration example prints minimum positions in both metres and
micrometres, Hessian eigenvalues, mesh size, and solver/search diagnostics. See
`docs/NUMERICAL_ASSUMPTIONS.md` before interpreting the result quantitatively.

The verified milestone-two outputs are under `validation_results/milestone_2`.
The focused follow-up for the extra 60 µm candidate is under
`validation_results/milestone_2_extra_candidate`. It records every
Hessian-valid candidate before lowest-three selection, boundary and separation
metrics, Hessian-stencil sensitivity, local mesh perturbations, and diagnostic
plots. The evidence classifies the fourth candidate as a recovered-gradient
interpolation artifact at a mesh facet; the physical model is unchanged.

The parsed reference outputs are under `validation_results/milestone_3`. See
`docs/DATASET_FORMAT.md` for the exact schema and coordinate transformations, and
`docs/REFERENCE_ARTICLE_NOTES.md` for the limited article facts used here. The
article's eight-rod octupole is not assumed equivalent to this repository's
current four-electrode FEM geometry; that relationship must be established by
validation against the supplied data.

The default reference benchmark evaluates rows 1–10 with the real-scale geometry
and writes under `validation_results/milestone_5/single_variant`. Use
`--geometry demonstrator` to reproduce the old provisional geometry. Use
`--start-row` and `--end-row` for an inclusive range, or `--random-count` with
`--random-seed` for a reproducible subset. Every production row runs in a fresh
interpreter so Gmsh state cannot couple configurations.

`rf-trap-scale-validation` runs the milestone-five mesh and convention study and
writes all artifacts under `validation_results/milestone_5`. The verified run
used 2.0, 1.5, and 1.0 mm meshes for the identity convention, tested absolute
versus E1-relative input handling, tested all-positive versus diagnostic
checkerboard polarity, and screened the five non-identity E2–E4 permutations.

The best milestone-five case is all-positive and E1-relative with source E2 and
E3 exchanged (`FEM E1,E2,E3,E4 <- source E1,E3,E2,E4`) at 2.0 mm mesh. It
completes all ten rows with mean error 1.087 mm, but only five rows have exactly
three pre-selection Hessian-valid candidates and the maximum matched error is
5.982 mm. This is much closer than Milestone 4, but it is not consistent enough
to justify large synthetic dataset generation. See `docs/MILESTONE_5_RESULTS.md`.

`rf-trap-hypothesis-validation` runs the milestone-six staged diagnostic. It
screens rows 1--10 at 2.0 mm mesh over absolute/relative inputs and outputs, all
six E1-preserving source-numbering maps, all eight square-symmetry coordinate
transforms, fitted and physical-scale interpretations, and diagnostic voltage
models. It promotes the best three hypotheses to rows 1--50 and writes complete
CSV, Markdown, and plot artifacts under `validation_results/milestone_6`.

The best promoted diagnostic completes 50/50 rows but has exactly-three topology
in only 36/50, mean error 1.27046 mm, and maximum error 5.92593 mm. The fitted
one-electrode basis vector is within 0.02135% of all-positive, while alternate
binary polarities are worse or fail. The mismatch is therefore primarily
model-class/topology limited, not an unresolved global orientation or polarity
convention. See `docs/MILESTONE_6_RESULTS.md`. Synthetic generation remains
unsafe.

`rf-trap-fem-audit` runs the milestone-seven numerical audit. It validates the
shared P1 Dirichlet solver against a concentric circular capacitor and an exact
linear field, checks `E=-grad(phi)`, audits every circular boundary marker and
real-scale geometry clearance, checks undisplaced D4 symmetry, flags or
audit-filters recovered-gradient artifacts, and reruns reference rows 1--10 at
2.0, 1.5, 1.0, and 0.75 mm meshes (plus rows 1--3 at 0.5 mm by default).

The analytic, sign, boundary, geometry, and 1 mm symmetry audits pass, and no
core FEM bug was found. Reference error improves only 0.215% from 2.0 to
0.75 mm and topology remains non-monotone. Because 4/30 selected h=2 mm minima
meet the conservative recovered-gradient artifact flag, the physical model is
likely incomplete but the residual is not yet scientifically attributable to
model class alone. See `docs/MILESTONE_7_RESULTS.md`. Synthetic generation
remains unsafe.

`rf-trap-robust-minima-validation` runs the milestone-eight post-processing
study. Robust mode combines the legacy candidate source, exact zeros of the
continuous recovered P1 field inside triangles, and local lows of the raw
element field. It audits internal-facet distance, adjacent raw-field jumps,
four mesh-scaled Hessian stencils, recovered-field scale, source support, and
accept/reject reasons. Rejected candidates are retained in the output tables.

For rows 1--10 at h=2 mm with the best E1,E3,E2,E4 mapping, robust selection
improves exactly-three topology from 5/10 to 10/10, rejects 24 extra candidates,
and selects no multi-stencil-unstable candidates. Mean reference error changes
only from 1.08687 to 1.08754 mm, so the old selected-minimum artifacts do not
explain a significant part of the mismatch. Robust topology remains exactly
three at 1.5, 1.0, and 0.75 mm; all 69 branch transitions at h<=1.5 mm are
stable within 0.25 mm. See `docs/MILESTONE_8_RESULTS.md`. The validation gate
still fails and synthetic generation remains unsafe.

This milestone intentionally contains no ML, inverse model, or synthetic dataset
generator.
