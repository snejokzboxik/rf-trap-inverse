# RF trap forward model and validation — milestone 2

This repository implements one reproducible 2D forward-model pipeline for four
infinitely long, equal-phase cylindrical electrodes.  Given the six Cartesian
displacements of electrodes 2–4, it:

1. builds a circular vacuum domain with four circular holes;
2. generates a conforming first-order triangular mesh with Gmsh;
3. solves Laplace's equation with scikit-fem;
4. recovers a continuous electric-field surrogate and evaluates `|E|²`;
5. coarse-scans, refines, merges, Hessian-validates, and polar-angle-sorts three
   local minima.

The package also runs full mesh-size × outer-radius convergence studies, compares
successive minima with minimum-distance spatial assignment, and writes CSV,
Markdown, and PNG reports.

The package uses metres and volts internally. Geometry and numerical values are
configuration inputs; the dimensions in `examples/run_one_configuration.py` are
only a demonstrator because the physical nominal geometry has not yet been
supplied.

## Install and run

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest
.venv\Scripts\python examples\run_one_configuration.py
.venv\Scripts\rf-trap-convergence
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
This milestone intentionally contains no ML, inverse model, or dataset generator.
