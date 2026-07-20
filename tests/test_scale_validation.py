"""Tests for milestone-5 case summarization using mocked forward data."""

from __future__ import annotations

import numpy as np

from rf_trap_forward.dataset import ReferenceDataset
from rf_trap_forward.real_scale import real_scale_forward_config
from rf_trap_forward.reference_validation import ForwardObservation, run_reference_validation
from rf_trap_forward.scale_validation import ScaleValidationCase, select_best_case


def _dataset() -> ReferenceDataset:
    return ReferenceDataset(
        np.zeros((1, 4, 2), dtype=float),
        np.asarray([[[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]]) * 1.0e-3,
    )


def _observation(positions_m: np.ndarray) -> ForwardObservation:
    return ForwardObservation(
        minima_positions_m=positions_m,
        hessian_validated_candidates=3,
        node_count=100,
        triangle_count=180,
        relative_free_residual=1.0e-14,
        valid_coarse_points=50,
        coarse_candidates=3,
        refined_candidates=3,
        unique_candidates=3,
    )


def test_best_case_prefers_completion_then_lower_mocked_error() -> None:
    """A lower-error complete case must win the comparable case summary."""

    dataset = _dataset()
    reference = dataset.minima_relative_to_electrode1_m[0]
    config = real_scale_forward_config(mesh_size_m=2.0e-3)
    worse = run_reference_validation(
        dataset,
        config,
        (1,),
        runner=lambda _displacements, _config: _observation(reference + 0.5e-3),
    )
    better = run_reference_validation(
        dataset,
        config,
        (1,),
        runner=lambda _displacements, _config: _observation(reference + 0.1e-3),
    )
    selected = select_best_case(
        (
            ScaleValidationCase("worse", "full", worse),
            ScaleValidationCase("better", "full", better),
        ),
        (1,),
    )
    assert selected.name == "better"
