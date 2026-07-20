# Reference article notes

Source: *Symmetry breaking in linear multipole traps*, arXiv:1705.08133v2
(2017), supplied locally as `1705.08133v2.pdf`.

Only the following facts are used by this project:

- The time-averaged ponderomotive pseudopotential is proportional to the squared
  RF electric-field magnitude, `|E_rf|²` (article page 4).
- Symmetry breaking lifts the degenerate RF-field zero of the ideal multipole and
  separates it into distinct zero-field positions (pages 4–7).
- For a multipole made from `2k` rods, the transverse RF field is described by a
  polynomial of degree `k - 1`; after symmetry breaking there are `k - 1`
  RF-field zero positions (pages 4 and 7).
- An octupole has `k = 4`, so three local pseudopotential minima are expected
  (pages 3–4 and 7).

## Model-equivalence warning

The article analyzes an octupole multipole trap made from eight rods with
alternating RF polarity. The current FEM implementation uses four equal-phase
cylindrical electrodes and a grounded circular outer boundary. These are not
assumed to be physically equivalent.

Consequently, the article motivates the expectation of a three-minimum structure
but does not validate the current four-electrode geometry. The FEM model must be
compared directly with the supplied reference dataset before any equivalence or
quantitative agreement is claimed. Milestone 3 deliberately performs only the
format verification and ingestion needed for that later validation; it does not
change the physical model.
