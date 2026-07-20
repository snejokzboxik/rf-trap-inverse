# RF trap forward model — milestone 1

This repository implements one reproducible 2D forward-model pipeline for four
infinitely long, equal-phase cylindrical electrodes.  Given the six Cartesian
displacements of electrodes 2–4, it:

1. builds a circular vacuum domain with four circular holes;
2. generates a conforming first-order triangular mesh with Gmsh;
3. solves Laplace's equation with scikit-fem;
4. recovers a continuous electric-field surrogate and evaluates `|E|²`;
5. coarse-scans, refines, merges, Hessian-validates, and polar-angle-sorts three
   local minima.

The package uses metres and volts internally.  Geometry and numerical values are
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
```

The example prints minimum positions in both metres and micrometres, Hessian
eigenvalues, mesh size, and solver/search diagnostics.  See
`docs/NUMERICAL_ASSUMPTIONS.md` before interpreting the result quantitatively.

This milestone intentionally contains no ML, inverse model, or dataset generator.

