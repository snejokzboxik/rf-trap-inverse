"""Mocked tests for Milestone-6 hypothesis transforms and ranking."""

from __future__ import annotations

import numpy as np
import pytest

from rf_trap_forward.dataset import ReferenceDataset
from rf_trap_forward.hypothesis_validation import (
    FEMHypothesis,
    OutputHypothesis,
    apply_coordinate_transform,
    build_hypothesis_result,
    fit_global_output_scale,
    passes_validation_gate,
    rank_hypotheses,
)
from rf_trap_forward.real_scale import real_scale_forward_config
from rf_trap_forward.reference_validation import (
    ForwardObservation,
    ReferenceValidationVariant,
    prepare_reference_row_inputs,
    run_reference_validation,
)


def _dataset() -> ReferenceDataset:
    displacements = np.asarray(
        [[[1.0, 2.0], [4.0, 8.0], [-2.0, 5.0], [7.0, -1.0]]],
        dtype=float,
    ) * 1.0e-4
    minima = np.asarray(
        [[[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]],
        dtype=float,
    ) * 1.0e-3
    return ReferenceDataset(displacements, minima)


def _observation(points_m: np.ndarray) -> ForwardObservation:
    return ForwardObservation(
        minima_positions_m=points_m,
        hessian_validated_candidates=3,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-14,
        valid_coarse_points=50,
        coarse_candidates=3,
        refined_candidates=3,
        unique_candidates=3,
    )


def test_coordinate_transforms_cover_rotation_sign_and_swap() -> None:
    """Named transforms must implement the documented global conventions."""

    points = np.asarray([[2.0, 3.0], [-1.0, 4.0]])
    np.testing.assert_allclose(
        apply_coordinate_transform(points, "flip-x"),
        [[-2.0, 3.0], [1.0, 4.0]],
    )
    np.testing.assert_allclose(
        apply_coordinate_transform(points, "swap-xy"),
        [[3.0, 2.0], [4.0, -1.0]],
    )
    np.testing.assert_allclose(
        apply_coordinate_transform(points, "rotate-90"),
        [[-3.0, 2.0], [-4.0, -1.0]],
    )


def test_e1_preserving_permutation_reorders_solver_displacements() -> None:
    """The FEM-slot map must reorder E2--E4 while retaining source E1."""

    dataset = _dataset()
    variant = ReferenceValidationVariant(
        name="perm-1324",
        displacement_mode="electrode1-relative",
        electrode_permutation=(1, 3, 2, 4),
    )
    solver, reference, _ = prepare_reference_row_inputs(
        dataset.raw_displacements_m[0],
        dataset.raw_minima_absolute_m[0],
        real_scale_forward_config(),
        variant,
    )
    np.testing.assert_allclose(
        solver,
        np.asarray([-3.0, 3.0, 3.0, 6.0, 6.0, -3.0]) * 1.0e-4,
    )
    np.testing.assert_allclose(
        reference,
        dataset.minima_relative_to_electrode1_m[0],
    )


def test_global_scale_fit_recovers_known_factor_with_permuted_points() -> None:
    """Scale fitting must update assignment and recover a synthetic factor."""

    reference = np.asarray([[-2.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
    computed = (reference / 2.5)[[2, 0, 1]]
    assert fit_global_output_scale([reference], [computed]) == pytest.approx(2.5)


def test_hypothesis_ranking_uses_mocked_forward_errors() -> None:
    """Comparable complete hypotheses must rank by lower spatial error."""

    dataset = _dataset()
    reference = dataset.minima_relative_to_electrode1_m[0]
    config = real_scale_forward_config()
    fem = FEMHypothesis(
        name="mock-fem",
        family="all-positive",
        displacement_mode="electrode1-relative",
        electrode_permutation=(1, 2, 3, 4),
        electrode_potentials_v=(1.0, 1.0, 1.0, 1.0),
    )

    def result(offset_m: float, name: str):
        report = run_reference_validation(
            dataset,
            config,
            (1,),
            runner=lambda _displacements, _config: _observation(
                reference + np.asarray([offset_m, 0.0])
            ),
        )
        hypothesis = OutputHypothesis(
            name=name,
            family="mock",
            fem_hypothesis=fem,
            reference_frame="electrode1-relative",
            coordinate_transform="identity",
            scale_mode="none",
        )
        return build_hypothesis_result(
            dataset,
            report,
            hypothesis,
            scope="test",
        )

    better = result(0.1e-3, "better")
    worse = result(0.8e-3, "worse")
    assert rank_hypotheses((worse, better))[0].hypothesis.name == "better"


def test_validation_gate_requires_completion_topology_and_error_limits() -> None:
    """Every condition in the conservative gate must be independently required."""

    accepted = dict(
        selected_rows=10,
        completed_rows=10,
        exactly_three_rows=10,
        mean_error_m=0.20e-3,
        maximum_error_m=0.45e-3,
    )
    assert passes_validation_gate(**accepted)
    assert not passes_validation_gate(**(accepted | {"completed_rows": 9}))
    assert not passes_validation_gate(**(accepted | {"exactly_three_rows": 9}))
    assert not passes_validation_gate(**(accepted | {"mean_error_m": 0.251e-3}))
    assert not passes_validation_gate(**(accepted | {"maximum_error_m": 0.501e-3}))
