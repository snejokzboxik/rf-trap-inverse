# Milestone 2 convergence and validation results

Verification date: 2026-07-20  
Platform: Windows, Python 3.12.13

The numerical environment used Gmsh 4.15.2, NumPy 2.5.1, SciPy 1.18.0,
scikit-fem 12.0.2, Matplotlib 3.11.1, and pytest 9.1.1.

## Study definition

The unchanged provisional displacement vector was:

```text
(120, -80, -150, 110, 90, 160) µm
```

The full Cartesian study used:

- mesh characteristic lengths: 120, 80, and 60 µm;
- outer-boundary radii: 3.5, 4.0, and 5.0 mm;
- nine total forward solves;
- a 10 µm successive-coordinate reporting tolerance;
- fresh interpreter isolation for every case.

## Results

- Every run returned the required three angle-sorted minima with positive Hessian
  eigenvalues and successful L-BFGS-B status.
- Meshes ranged from 3,244 to 25,506 nodes and from 6,242 to 50,358 triangles.
- The largest relative free-node residual was `3.135e-15`.
- The largest matched displacement over any mesh-refinement step was 7.463 µm.
- The largest matched displacement over any outer-radius step was 8.330 µm.
- All selected-branch coordinate changes were below the configured 10 µm
  tolerance.

The finest mesh step (80 → 60 µm) had a maximum shift of approximately 5.046 µm
across the three outer radii. At the finest 60 µm mesh, the largest outer-radius
step was approximately 2.426 µm.

## Three-minimum stability conclusion

The strict three-minimum structure is **not yet validated**. At mesh size 60 µm
and outer radius 4.0 mm, coarse detection, refinement, duplicate merging, and
Hessian validation all retained four candidates. The milestone-one forward API
then selected the three lowest-`|E|²` candidates as designed.

Thus, the original strict candidate-count criterion remains false even though the
three selected branches are spatially consistent. The focused follow-up below
classifies the additional candidate and separates this numerical detection from
the physical-null interpretation.

## Focused follow-up on the fourth candidate

The follow-up investigation retained and reported all four pre-selection
Hessian-valid candidates for the 60 µm, 4.0 mm case. The unselected candidate is
at `(261.186943, -0.763932) µm` with `|E|² = 194.465258 V²/m²`, approximately
245,789 times the largest value among the three selected null-like candidates.

It is not a search/electrode-boundary artifact: its nearest electrode-surface and
search-window clearances are 518.813 and 438.813 µm. It is not a duplicate: its
nearest other candidate is 225.322 µm away, compared with the 20 µm merge
threshold.

The point is only 0.0111 µm from an internal mesh facet. Its finite-difference
Hessian eigenvalues vary strongly with stencil size and the smaller eigenvalue
becomes negative at a 32 µm step. The extra point is absent when the mesh size is
perturbed to either 59 or 61 µm. The combined evidence classifies it as a
**recovered-gradient interpolation artifact at a mesh facet**, admitted by the
finite-difference Hessian test, rather than a fourth physical RF null.

The focused tables and plots are in
`validation_results/milestone_2_extra_candidate/extra_candidate_report.md`.
No physical equation, boundary condition, or selection rule was changed.

## Reproducibility correction

Validation exposed hidden Gmsh history and launch-context inputs. The mesher now:

- passes an empty argument list and disables external Gmsh configuration files;
- uses explicit OpenCASCADE primitive tags;
- requires a positive seed and exposes the random factor;
- enables Gmsh reproducibility mode;
- isolates production convergence cases in fresh interpreter processes.

These changes do not alter the physical model.

## Verification artifacts

The complete outputs are in `validation_results/milestone_2`:

- `convergence_runs.csv`;
- `coordinate_comparisons.csv`;
- `convergence_report.md`;
- `mesh_refinement_minima.png`;
- `outer_radius_minima.png`.

Eighteen tests pass, including synthetic/mocked convergence and candidate-
classification logic plus headless artifact generation. No ML, inverse model,
or bulk dataset generation was added.
