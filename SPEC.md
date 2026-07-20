\# Inverse geometry reconstruction for a synchronous RF quadrupole trap



\## Goal



Develop a research-grade Python framework for solving both the forward and inverse problems of a 2D four-electrode synchronous RF trap.



The project is independent from any previous repository.



\---



\## Physics



Model a 2D cross-section of four infinitely long cylindrical electrodes.



All electrodes have identical RF phase:



V = +1 (normalized).



Outer boundary approximates infinity:



φ = 0.



Solve



∇²φ = 0



using FEM.



Compute



E = -∇φ



and



Ψ = |E|².



Ψ is proportional to the RF pseudopotential, therefore only its local minima are required.



\---



\## Geometry



Electrode #1 is the reference frame.



Unknown parameters are



(dx2, dy2,

&#x20;dx3, dy3,

&#x20;dx4, dy4)



generated independently from



U(-200 μm, 200 μm).



Electrode radius and nominal geometry will be supplied later and therefore must come from configuration.



\---



\## Forward problem



Input:



6 electrode displacement coordinates.



Output:



3 local minima



(x1,y1),

(x2,y2),

(x3,y3)



sorted by polar angle.



\---



\## Numerical requirements



Use Python.



Prefer:



\- scikit-fem

\- gmsh/pygmsh if needed

\- numpy

\- scipy

\- matplotlib

\- pytest



Minima search:



1\. coarse scan

2\. candidate detection

3\. L-BFGS-B refinement

4\. merge duplicates

5\. Hessian validation



Reject minima inside electrodes.



\---



\## Dataset



Each sample stores



dx2 dy2 dx3 dy3 dx4 dy4

x1 y1 x2 y2 x3 y3



Additional diagnostics are encouraged.



\---



\## Code quality



Research-grade architecture.



Typed code.



Dataclasses.



Unit tests.



Clear separation of:



geometry



mesh



solver



field



minima



dataset



validation



No notebooks.



No hardcoded constants.



Everything configurable.



\---



\## Current objective



Implement ONLY the forward solver.



Do not implement machine learning yet.



Accuracy, reproducibility and clean architecture have priority over speed.



Whenever an implementation choice is unclear, explain the tradeoffs before coding.

