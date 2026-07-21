"""Pure matching tests for the focused row-5 diagnostics."""

from __future__ import annotations

import numpy as np

from rf_trap_forward.reference_validation import match_minima_by_distance
from rf_trap_forward.row5_debug import (
    enumerate_three_point_assignments,
    pairwise_distance_matrix_m,
)


def test_pairwise_distance_matrix_has_reference_rows_and_computed_columns() -> None:
    reference = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 3.0)))
    computed = np.asarray(((0.0, 3.1), (0.1, 0.0), (2.2, 0.0)))
    matrix = pairwise_distance_matrix_m(reference, computed)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix[0], (3.1, 0.1, 2.2))


def test_all_six_assignments_are_ranked_and_match_hungarian_optimum() -> None:
    reference = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 3.0)))
    computed = np.asarray(((0.0, 3.1), (0.1, 0.0), (2.2, 0.0)))
    matrix = pairwise_distance_matrix_m(reference, computed)
    options = enumerate_three_point_assignments(matrix)
    assert len(options) == 6
    assert options[0].computed_indices == (2, 3, 1)
    matches = match_minima_by_distance(reference, computed)
    assert tuple(item.computed_index for item in matches) == options[0].computed_indices
    np.testing.assert_allclose(
        [item.distance_m for item in matches],
        options[0].errors_m,
    )


def test_direct_nearest_neighbor_can_be_compared_without_resolving_fem() -> None:
    distances = np.asarray(
        ((0.1, 2.0, 3.0), (0.2, 2.2, 3.2), (4.0, 0.3, 0.4))
    )
    nearest = tuple(int(value) + 1 for value in np.argmin(distances, axis=1))
    assert nearest == (1, 1, 2)
    assert len(set(nearest)) < 3
    assert enumerate_three_point_assignments(distances)[0].computed_indices != nearest
