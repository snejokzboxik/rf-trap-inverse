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

Thus, the three selected branches remain spatially consistent across the sweep,
but one recovered-field realization contains an additional positive-Hessian local
minimum. This may be a gradient-recovery or finite-difference artifact and must be
investigated before claiming stable physical three-minimum topology.

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

Fourteen tests pass, including synthetic/mocked convergence logic and headless
artifact generation. No ML, inverse model, or bulk dataset generation was added.
