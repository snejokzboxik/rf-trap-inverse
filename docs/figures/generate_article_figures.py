"""Generate publication-ready article figures from verified project artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from numpy.typing import NDArray

from rf_trap_forward.absolute_validation import (
    wolfram_to_fem_absolute_displacements_m,
)
from rf_trap_forward.dataset import sort_points_by_polar_angle
from rf_trap_forward.field import RecoveredField, recover_field
from rf_trap_forward.geometry import (
    TrapGeometry,
    build_geometry_from_absolute_displacements,
)
from rf_trap_forward.mesh import TrapMesh, generate_mesh
from rf_trap_forward.minima_modes import RobustMinimaConfig, run_minima_mode
from rf_trap_forward.real_scale import DIAGONAL_ALTERNATING_POTENTIALS_V
from rf_trap_forward.solver import FEMSolution, solve_potential
from rf_trap_forward.synthetic_dataset import practical_generator_forward_config


INK = "#20252B"
BLUE = "#2F6B8A"
BLUE_LIGHT = "#DCEAF1"
ORANGE = "#D97706"
RED = "#C63D4F"
GREY = "#89939D"
GREY_LIGHT = "#D9DEE3"
GRID = "#D7DCE1"
WHITE = "#FFFFFF"
MICROMETRES_PER_METRE = 1.0e6

LEARNING_ROWS = np.asarray((1000, 5000, 10000, 20000, 29995), dtype=int)
LEARNING_MAE_UM = np.asarray(
    (120.805389, 108.673243, 105.740198, 104.563974, 103.613591)
)
LEARNING_RMSE_UM = np.asarray(
    (150.065385, 138.338341, 134.646840, 133.533373, 132.388747)
)
LEARNING_MAX_UM = np.asarray(
    (580.788191, 690.648183, 538.559993, 535.056322, 549.919969)
)

MODEL_NAMES = ("Ridge", "Random Forest", "MLP")
MODEL_MAE_UM = np.asarray((170.616817, 129.513125, 102.891015))
MODEL_RMSE_UM = np.asarray((209.663012, 160.499889, 132.217767))
MODEL_MAX_UM = np.asarray((985.042242, 638.749610, 530.827205))


@dataclass(frozen=True)
class RepresentativeSample:
    """One clean merged-dataset row used for the numerical FEM figure."""

    sample_id: int
    row_index: int
    wolfram_displacements_m: NDArray[np.float64]
    fem_displacements_m: NDArray[np.float64]
    recorded_minima_m: NDArray[np.float64]


@dataclass(frozen=True)
class ConnectionResult:
    """One electrical connection solved on the shared displaced mesh."""

    name: str
    potentials_v: tuple[float, float, float, float]
    geometry: TrapGeometry
    solution: FEMSolution
    recovered_field: RecoveredField
    minima_m: NDArray[np.float64]


@dataclass(frozen=True)
class FEMComparison:
    """Shared mesh and numerical products for the two connection definitions."""

    sample: RepresentativeSample
    trap_mesh: TrapMesh
    quadrupole: ConnectionResult
    in_phase: ConnectionResult


def configure_matplotlib() -> None:
    """Set one restrained, publication-oriented Matplotlib style."""

    mpl.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure_formats(figure: mpl.figure.Figure, output_base: Path) -> list[Path]:
    """Save one figure as high-resolution PNG plus vector PDF and SVG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".svg"),
    ]
    figure.savefig(paths[0], dpi=320, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[1], bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[2], bbox_inches="tight", pad_inches=0.04)
    return paths


def load_representative_sample(
    dataset_path: Path,
    *,
    row_index: int = 0,
) -> RepresentativeSample:
    """Load one deterministic clean sample without modifying the source dataset."""

    if row_index < 0:
        raise ValueError("row_index must be non-negative")
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        selected: dict[str, str] | None = None
        for index, row in enumerate(reader):
            if index == row_index:
                selected = row
                break
    if selected is None:
        raise ValueError(f"dataset does not contain row index {row_index}")
    wolfram_columns = [
        f"w{electrode}_{component}_m"
        for electrode in range(1, 5)
        for component in ("dx", "dy")
    ]
    minima_columns = [
        f"min{minimum}_{component}_m"
        for minimum in range(1, 4)
        for component in ("x", "y")
    ]
    wolfram = np.asarray(
        [float(selected[column]) for column in wolfram_columns], dtype=float
    ).reshape(4, 2)
    minima = np.asarray(
        [float(selected[column]) for column in minima_columns], dtype=float
    ).reshape(3, 2)
    if not np.all(np.isfinite(wolfram)) or not np.all(np.isfinite(minima)):
        raise ValueError("selected clean row contains a non-finite value")
    return RepresentativeSample(
        sample_id=int(selected["sample_id"]),
        row_index=row_index,
        wolfram_displacements_m=wolfram,
        fem_displacements_m=wolfram_to_fem_absolute_displacements_m(wolfram),
        recorded_minima_m=minima,
    )


def solve_connection_comparison(sample: RepresentativeSample) -> FEMComparison:
    """Solve checkerboard and in-phase connections on one identical FEM mesh."""

    base_config = practical_generator_forward_config("practical")
    quadrupole_geometry_config = replace(
        base_config.geometry,
        electrode_potentials_v=DIAGONAL_ALTERNATING_POTENTIALS_V,
    )
    in_phase_geometry = build_geometry_from_absolute_displacements(
        base_config.geometry,
        sample.fem_displacements_m,
    )
    quadrupole_geometry = build_geometry_from_absolute_displacements(
        quadrupole_geometry_config,
        sample.fem_displacements_m,
    )
    trap_mesh = generate_mesh(in_phase_geometry, base_config.mesh)
    in_phase_solution = solve_potential(
        in_phase_geometry,
        trap_mesh,
        base_config.solver,
    )
    quadrupole_solution = solve_potential(
        quadrupole_geometry,
        trap_mesh,
        base_config.solver,
    )
    in_phase_field = recover_field(in_phase_solution)
    quadrupole_field = recover_field(quadrupole_solution)
    robust_config = RobustMinimaConfig()
    in_phase_mode = run_minima_mode(
        in_phase_field,
        in_phase_solution.potential_v,
        base_config.minima,
        mode="robust",
        robust_config=robust_config,
    )
    quadrupole_search = replace(base_config.minima, expected_minima=1)
    quadrupole_mode = run_minima_mode(
        quadrupole_field,
        quadrupole_solution.potential_v,
        quadrupole_search,
        mode="robust",
        robust_config=robust_config,
    )
    in_phase_minima = _ordered_minima(in_phase_mode.minima)
    quadrupole_minima = _ordered_minima(quadrupole_mode.minima)
    if in_phase_minima.shape != (3, 2):
        raise RuntimeError(
            "selected in-phase case must contain exactly three robust minima"
        )
    if not np.array_equal(in_phase_minima, sample.recorded_minima_m):
        maximum_error_um = MICROMETRES_PER_METRE * float(
            np.max(
                np.linalg.norm(
                    in_phase_minima - sample.recorded_minima_m,
                    axis=1,
                )
            )
        )
        raise RuntimeError(
            "in-phase solve does not reproduce the stored clean minima; "
            f"maximum ordered error is {maximum_error_um:.9g} um"
        )
    return FEMComparison(
        sample=sample,
        trap_mesh=trap_mesh,
        quadrupole=ConnectionResult(
            name="Quadrupole checkerboard",
            potentials_v=DIAGONAL_ALTERNATING_POTENTIALS_V,
            geometry=quadrupole_geometry,
            solution=quadrupole_solution,
            recovered_field=quadrupole_field,
            minima_m=quadrupole_minima,
        ),
        in_phase=ConnectionResult(
            name="In-phase",
            potentials_v=(1.0, 1.0, 1.0, 1.0),
            geometry=in_phase_geometry,
            solution=in_phase_solution,
            recovered_field=in_phase_field,
            minima_m=in_phase_minima,
        ),
    )


def _ordered_minima(minima: Sequence[object]) -> NDArray[np.float64]:
    """Return existing minima in the canonical polar-angle order."""

    if not minima:
        return np.empty((0, 2), dtype=float)
    positions = np.vstack([np.asarray(item.position_m, dtype=float) for item in minima])
    return sort_points_by_polar_angle(positions)


def create_concept_figure(output_base: Path) -> list[Path]:
    """Draw the six-stage inverse-reconstruction pipeline as a vector schematic."""

    figure, axes = plt.subplots(1, 6, figsize=(14.8, 3.3))
    titles = (
        "Displaced\nelectrodes",
        "Forward FEM\nmodel",
        "Three minima\nextraction",
        "Supervised\ndataset",
        "Inverse MLP\nmodel",
        "Reconstructed\ndisplacements",
    )
    for index, (axis, title) in enumerate(zip(axes, titles, strict=True)):
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.text(
            -0.98,
            0.98,
            f"({chr(97 + index)})",
            ha="left",
            va="top",
            color=INK,
            weight="semibold",
        )
        axis.set_title(title, pad=5)
    _draw_displaced_geometry(axes[0], reconstructed=False)
    _draw_regular_mesh_schematic(axes[1])
    _draw_three_minima_schematic(axes[2])
    _draw_dataset_schematic(axes[3])
    _draw_mlp_schematic(axes[4])
    _draw_displaced_geometry(axes[5], reconstructed=True)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.10, top=0.83, wspace=0.34)
    for left, right in zip(axes[:-1], axes[1:], strict=True):
        start = figure.transFigure.inverted().transform(
            left.transAxes.transform((1.02, 0.50))
        )
        end = figure.transFigure.inverted().transform(
            right.transAxes.transform((-0.02, 0.50))
        )
        arrow = FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=INK,
            clip_on=False,
        )
        figure.add_artist(arrow)
    figure.text(
        0.5,
        0.025,
        "Synthetic forward map: 8 electrode-displacement coordinates "
        r"$\rightarrow$ 6 minimum coordinates; inverse regression reconstructs 8 coordinates",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=8,
    )
    paths = save_figure_formats(figure, output_base)
    plt.close(figure)
    return paths


def _draw_displaced_geometry(
    axis: mpl.axes.Axes,
    *,
    reconstructed: bool,
) -> None:
    """Draw a compact four-electrode displacement schematic."""

    nominal = np.asarray(
        ((-0.50, 0.50), (0.50, 0.50), (-0.50, -0.50), (0.50, -0.50))
    )
    offsets = np.asarray(
        ((0.08, 0.03), (-0.03, 0.08), (-0.07, -0.02), (0.04, -0.07))
    )
    axis.add_patch(Circle((0.0, 0.0), 0.67, fill=False, ls="--", lw=0.8, ec=GREY))
    for index, (base, offset) in enumerate(
        zip(nominal, offsets, strict=True), start=1
    ):
        displaced = base + offset
        axis.add_patch(Circle(base, 0.13, fill=False, ls=":", lw=0.8, ec=GREY))
        axis.add_patch(
            Circle(
                displaced,
                0.13,
                facecolor=BLUE_LIGHT if reconstructed else WHITE,
                edgecolor=INK,
                linewidth=1.0,
            )
        )
        axis.annotate(
            "",
            xy=displaced + 1.5 * offset,
            xytext=displaced,
            arrowprops={
                "arrowstyle": "-|>",
                "color": BLUE if reconstructed else ORANGE,
                "lw": 1.2,
            },
        )
        axis.text(
            displaced[0],
            displaced[1],
            f"{index}",
            ha="center",
            va="center",
            fontsize=7,
            color=INK,
        )
    label = r"$\hat{\mathbf{d}}\in\mathbb{R}^{8}$" if reconstructed else r"$\mathbf{d}\in\mathbb{R}^{8}$"
    axis.text(0.0, -0.91, label, ha="center", va="bottom", color=INK)


def _draw_regular_mesh_schematic(axis: mpl.axes.Axes) -> None:
    """Draw an intentionally regular triangular mesh clipped to a disk."""

    vertical_step = 0.16 * np.sqrt(3.0) / 2.0
    points: list[tuple[float, float]] = []
    y_values = np.arange(-0.82, 0.83, vertical_step)
    for row_index, y_value in enumerate(y_values):
        shift = 0.08 if row_index % 2 else 0.0
        for x_value in np.arange(-0.88, 0.89, 0.16):
            point = (x_value + shift, y_value)
            if np.hypot(*point) <= 0.84:
                points.append(point)
    coordinates = np.asarray(points)
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    centroids = coordinates[triangulation.triangles].mean(axis=1)
    triangulation.set_mask(np.linalg.norm(centroids, axis=1) > 0.82)
    axis.triplot(triangulation, color=GREY, linewidth=0.32)
    axis.add_patch(Circle((0.0, 0.0), 0.84, fill=False, ec=INK, lw=1.0))
    for center in ((-0.48, 0.48), (0.48, 0.48), (-0.48, -0.48), (0.48, -0.48)):
        axis.add_patch(Circle(center, 0.15, facecolor=WHITE, edgecolor=INK, lw=0.9))
    axis.text(
        0.0,
        -0.96,
        r"$\nabla^2\phi=0,\quad \mathbf{E}=-\nabla\phi$",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )


def _draw_three_minima_schematic(axis: mpl.axes.Axes) -> None:
    """Draw three and only three schematic pseudopotential minima."""

    centers = np.asarray(((0.0, 0.48), (-0.45, -0.27), (0.45, -0.27)))
    for radius, alpha in ((0.38, 0.35), (0.27, 0.50), (0.16, 0.75)):
        for center in centers:
            axis.add_patch(
                Circle(
                    center,
                    radius,
                    fill=False,
                    edgecolor=BLUE,
                    linewidth=0.75,
                    alpha=alpha,
                )
            )
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        s=34,
        c=RED,
        edgecolors=INK,
        linewidths=0.6,
        zorder=5,
    )
    axis.text(
        0.0,
        -0.94,
        r"$\Psi\propto|\mathbf{E}|^2$",
        ha="center",
        va="bottom",
        color=INK,
    )


def _draw_dataset_schematic(axis: mpl.axes.Axes) -> None:
    """Draw the 6-to-8 supervised table relationship."""

    axis.add_patch(
        FancyBboxPatch(
            (-0.82, -0.64),
            1.64,
            1.30,
            boxstyle="round,pad=0.04",
            facecolor=WHITE,
            edgecolor=INK,
            linewidth=0.9,
        )
    )
    axis.add_patch(Rectangle((-0.70, 0.35), 0.61, 0.18, facecolor=BLUE_LIGHT, edgecolor=INK, lw=0.6))
    axis.add_patch(Rectangle((0.09, 0.35), 0.61, 0.18, facecolor="#F8E7D0", edgecolor=INK, lw=0.6))
    axis.text(-0.40, 0.44, "X: 6 minima", ha="center", va="center", fontsize=7)
    axis.text(0.40, 0.44, "Y: 8 shifts", ha="center", va="center", fontsize=7)
    for row in range(4):
        y_value = 0.20 - row * 0.19
        axis.plot((-0.70, -0.09), (y_value, y_value), color=GREY, lw=3.0, solid_capstyle="butt")
        axis.plot((0.09, 0.70), (y_value, y_value), color=GREY, lw=3.0, solid_capstyle="butt")
    axis.text(
        0.0,
        -0.92,
        "Clean FEM samples",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )


def _draw_mlp_schematic(axis: mpl.axes.Axes) -> None:
    """Draw a compact multilayer perceptron schematic."""

    layer_x = (-0.72, -0.24, 0.24, 0.72)
    layer_sizes = (5, 4, 4, 5)
    positions: list[list[tuple[float, float]]] = []
    for x_value, layer_size in zip(layer_x, layer_sizes, strict=True):
        y_values = np.linspace(-0.58, 0.58, layer_size)
        positions.append([(x_value, float(y_value)) for y_value in y_values])
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        for start in left:
            for end in right:
                axis.plot(
                    (start[0], end[0]),
                    (start[1], end[1]),
                    color=GREY_LIGHT,
                    linewidth=0.35,
                    zorder=1,
                )
    for layer_index, layer in enumerate(positions):
        facecolor = BLUE_LIGHT if layer_index in (0, 3) else WHITE
        for position in layer:
            axis.add_patch(
                Circle(position, 0.065, facecolor=facecolor, edgecolor=INK, lw=0.8, zorder=2)
            )
    axis.text(
        0.0,
        -0.93,
        r"$f_\theta:\mathbb{R}^{6}\rightarrow\mathbb{R}^{8}$",
        ha="center",
        va="bottom",
        color=INK,
    )


def create_fem_connections_figure(
    comparison: FEMComparison,
    output_base: Path,
) -> list[Path]:
    """Plot the shared real mesh and real effective-potential proxy for both drives."""

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 9.1))
    _plot_mesh_panel(
        axes[0, 0],
        comparison.trap_mesh,
        comparison.quadrupole.geometry,
        comparison.quadrupole.potentials_v,
        "(a) Quadrupole connection: FEM mesh",
    )
    _plot_effective_potential_panel(
        axes[0, 1],
        comparison.trap_mesh,
        comparison.quadrupole,
        "(b) Quadrupole connection: effective potential",
    )
    _plot_mesh_panel(
        axes[1, 0],
        comparison.trap_mesh,
        comparison.in_phase.geometry,
        comparison.in_phase.potentials_v,
        "(c) In-phase connection: FEM mesh",
    )
    _plot_effective_potential_panel(
        axes[1, 1],
        comparison.trap_mesh,
        comparison.in_phase,
        "(d) In-phase connection: effective potential",
    )
    figure.suptitle(
        f"Real 2D FEM comparison for one displaced geometry "
        f"(sample {comparison.sample.sample_id})",
        fontsize=12,
        weight="semibold",
        color=INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Shared practical mesh: 500 µm central refinement; "
        r"effective-potential proxy $\Psi=|\mathbf{E}|^2$",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.96,
        bottom=0.07,
        top=0.94,
        hspace=0.24,
        wspace=0.21,
    )
    paths = save_figure_formats(figure, output_base)
    plt.close(figure)
    return paths


def _plot_mesh_panel(
    axis: mpl.axes.Axes,
    trap_mesh: TrapMesh,
    geometry: TrapGeometry,
    potentials_v: Sequence[float],
    title: str,
) -> None:
    """Plot an accurate crop of the shared locally refined triangular mesh."""

    points_mm = 1.0e3 * trap_mesh.mesh.p
    axis.triplot(
        points_mm[0],
        points_mm[1],
        trap_mesh.mesh.t.T,
        color="#77818A",
        linewidth=0.20,
        alpha=0.88,
        rasterized=False,
    )
    _draw_electrode_holes(axis, geometry, potentials_v)
    _format_geometry_axis(axis, title)
    axis.text(
        0.02,
        0.02,
        f"{trap_mesh.number_of_nodes:,} nodes\n"
        f"{trap_mesh.number_of_triangles:,} triangles",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK,
        bbox={"facecolor": WHITE, "edgecolor": GREY_LIGHT, "pad": 2.5},
        zorder=10,
    )


def _plot_effective_potential_panel(
    axis: mpl.axes.Axes,
    trap_mesh: TrapMesh,
    connection: ConnectionResult,
    title: str,
) -> None:
    """Plot the recovered nodal ``|E|^2`` field and robust minima."""

    points_mm = 1.0e3 * trap_mesh.mesh.p
    electric_field = connection.recovered_field.electric_field_nodes_v_per_m
    psi_nodes = np.einsum("ij,ij->j", electric_field, electric_field)
    crop_mask = (
        (np.abs(points_mm[0]) <= 27.0)
        & (np.abs(points_mm[1]) <= 27.0)
        & (psi_nodes > 0.0)
    )
    crop_values = psi_nodes[crop_mask]
    if crop_values.size == 0:
        raise RuntimeError("effective-potential plot crop contains no valid FEM nodes")
    lower = max(float(np.percentile(crop_values, 0.5)), np.finfo(float).tiny)
    upper = max(float(np.percentile(crop_values, 99.5)), 10.0 * lower)
    color_mesh = axis.tripcolor(
        points_mm[0],
        points_mm[1],
        trap_mesh.mesh.t.T,
        np.maximum(psi_nodes, lower),
        shading="gouraud",
        cmap="viridis",
        norm=LogNorm(vmin=lower, vmax=upper),
        rasterized=True,
    )
    _draw_electrode_holes(axis, connection.geometry, connection.potentials_v)
    if connection.minima_m.size:
        minima_mm = 1.0e3 * connection.minima_m
        axis.scatter(
            minima_mm[:, 0],
            minima_mm[:, 1],
            s=36,
            c=RED,
            edgecolors=WHITE,
            linewidths=0.9,
            marker="o",
            zorder=8,
        )
        axis.text(
            0.98,
            0.02,
            f"Robust minima: {len(minima_mm)}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=INK,
            bbox={"facecolor": WHITE, "edgecolor": GREY_LIGHT, "pad": 2.5},
            zorder=10,
        )
    _format_geometry_axis(axis, title)
    colorbar = axis.figure.colorbar(color_mesh, ax=axis, fraction=0.046, pad=0.025)
    colorbar.set_label(r"$\Psi=|\mathbf{E}|^2$ (V$^2$ m$^{-2}$)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _draw_electrode_holes(
    axis: mpl.axes.Axes,
    geometry: TrapGeometry,
    potentials_v: Sequence[float],
) -> None:
    """Overlay exact electrode positions and connection signs on a plot."""

    radius_mm = 1.0e3 * geometry.config.electrode_radius_m
    for center_m, potential_v in zip(
        geometry.centers_m,
        potentials_v,
        strict=True,
    ):
        center_mm = 1.0e3 * center_m
        axis.add_patch(
            Circle(
                center_mm,
                radius_mm,
                facecolor=WHITE,
                edgecolor=INK,
                linewidth=0.9,
                zorder=6,
            )
        )
        sign = "+" if potential_v > 0.0 else "−" if potential_v < 0.0 else "0"
        axis.text(
            center_mm[0],
            center_mm[1],
            sign,
            ha="center",
            va="center",
            fontsize=10,
            weight="semibold",
            color=INK,
            zorder=7,
        )


def _format_geometry_axis(axis: mpl.axes.Axes, title: str) -> None:
    """Apply common geometry limits, aspect ratio, labels, and title."""

    axis.set_xlim(-27.0, 27.0)
    axis.set_ylim(-27.0, 27.0)
    axis.set_aspect("equal")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title(title, loc="left", pad=6)
    axis.tick_params(direction="out", length=3)


def write_fem_metadata(
    comparison: FEMComparison,
    output_paths: Sequence[Path],
    metadata_path: Path,
) -> None:
    """Write provenance, connection definitions, and minima counts as JSON."""

    sample = comparison.sample
    metadata = {
        "coordinate_units": "metres",
        "effective_potential_proxy": "Psi = |E|^2 in V^2/m^2",
        "fem_electrode_order": [
            "F1 upper-left",
            "F2 upper-right",
            "F3 lower-left",
            "F4 lower-right",
        ],
        "mesh": {
            "central_mesh_size_m": 500.0e-6,
            "node_count": comparison.trap_mesh.number_of_nodes,
            "triangle_count": comparison.trap_mesh.number_of_triangles,
        },
        "outputs": [path.as_posix() for path in output_paths],
        "selected_row_index": sample.row_index,
        "selected_sample_id": sample.sample_id,
        "source_dataset": "merged N=51974 clean ML dataset",
        "true_displacements_fem_order_m": sample.fem_displacements_m.tolist(),
        "true_displacements_wolfram_order_m": (
            sample.wolfram_displacements_m.tolist()
        ),
        "connections": {
            "quadrupole": {
                "definition": "alternating checkerboard in FEM order",
                "electrode_potentials_v": list(
                    comparison.quadrupole.potentials_v
                ),
                "minimum_count": int(comparison.quadrupole.minima_m.shape[0]),
                "minima_coordinates_m": comparison.quadrupole.minima_m.tolist(),
                "relative_free_residual": (
                    comparison.quadrupole.solution.relative_free_residual
                ),
            },
            "in_phase": {
                "definition": "all four electrodes at the same RF phase",
                "electrode_potentials_v": list(comparison.in_phase.potentials_v),
                "minimum_count": int(comparison.in_phase.minima_m.shape[0]),
                "minima_coordinates_m": comparison.in_phase.minima_m.tolist(),
                "recorded_clean_minima_coordinates_m": (
                    sample.recorded_minima_m.tolist()
                ),
                "relative_free_residual": (
                    comparison.in_phase.solution.relative_free_residual
                ),
            },
        },
        "wolfram_to_fem_transform": "[-W3, -W1, -W4, -W2]",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_learning_metrics(metrics_path: Path) -> None:
    """Verify the article learning values against the tracked result table."""

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {
            int(row["requested_dataset_rows"]): row
            for row in csv.DictReader(stream)
            if row["model"] == "mlp"
        }
    observed_mae = np.asarray(
        [float(rows[int(size)]["overall_mae_um"]) for size in LEARNING_ROWS]
    )
    observed_rmse = np.asarray(
        [float(rows[int(size)]["overall_rmse_um"]) for size in LEARNING_ROWS]
    )
    observed_maximum = np.asarray(
        [float(rows[int(size)]["max_absolute_error_um"]) for size in LEARNING_ROWS]
    )
    checks = (
        (observed_mae, LEARNING_MAE_UM, "MAE"),
        (observed_rmse, LEARNING_RMSE_UM, "RMSE"),
        (observed_maximum, LEARNING_MAX_UM, "maximum absolute error"),
    )
    for observed, expected, label in checks:
        if not np.array_equal(np.round(observed, 6), expected):
            raise RuntimeError(f"tracked learning-curve {label} values changed")


def create_learning_curves_figure(output_base: Path) -> list[Path]:
    """Plot the three exact MLP learning-curve metrics."""

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.45))
    metrics = (
        ("(a) Mean absolute error", LEARNING_MAE_UM, "MAE (µm)", (98.0, 124.0)),
        ("(b) Root-mean-square error", LEARNING_RMSE_UM, "RMSE (µm)", (128.0, 153.0)),
        (
            "(c) Maximum absolute error",
            LEARNING_MAX_UM,
            "Maximum error (µm)",
            (500.0, 710.0),
        ),
    )
    for axis, (title, values, ylabel, limits) in zip(axes, metrics, strict=True):
        axis.plot(
            LEARNING_ROWS,
            values,
            color=BLUE,
            linewidth=1.7,
            marker="o",
            markersize=4.8,
            markerfacecolor=WHITE,
            markeredgewidth=1.2,
        )
        for x_value, y_value in zip(LEARNING_ROWS, values, strict=True):
            axis.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Dataset rows used")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.set_xticks(LEARNING_ROWS)
        axis.set_xticklabels(("1k", "5k", "10k", "20k", "29,995"))
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "MLP learning curves on fixed evaluation data",
        fontsize=12,
        weight="semibold",
        color=INK,
        y=1.02,
    )
    figure.tight_layout(w_pad=1.8)
    paths = save_figure_formats(figure, output_base)
    plt.close(figure)
    return paths


def validate_model_metrics(metrics_path: Path) -> None:
    """Verify the N=51974 article metrics against the tracked training table."""

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["model"]: row for row in csv.DictReader(stream)}
    names = ("ridge", "random_forest", "mlp")
    observed_mae = np.asarray([float(rows[name]["overall_mae_um"]) for name in names])
    observed_rmse = np.asarray(
        [float(rows[name]["overall_rmse_um"]) for name in names]
    )
    observed_maximum = np.asarray(
        [float(rows[name]["max_absolute_error_um"]) for name in names]
    )
    checks = (
        (observed_mae, MODEL_MAE_UM, "MAE"),
        (observed_rmse, MODEL_RMSE_UM, "RMSE"),
        (observed_maximum, MODEL_MAX_UM, "maximum absolute error"),
    )
    for observed, expected, label in checks:
        if not np.array_equal(np.round(observed, 6), expected):
            raise RuntimeError(f"tracked N=51974 model {label} values changed")


def create_model_comparison_figure(output_base: Path) -> list[Path]:
    """Plot zero-based model comparisons for three regression metrics."""

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.55))
    colors = (GREY_LIGHT, ORANGE, BLUE)
    metrics = (
        ("(a) Mean absolute error", MODEL_MAE_UM, "MAE (µm)"),
        ("(b) Root-mean-square error", MODEL_RMSE_UM, "RMSE (µm)"),
        ("(c) Maximum absolute error", MODEL_MAX_UM, "Maximum error (µm)"),
    )
    for axis, (title, values, ylabel) in zip(axes, metrics, strict=True):
        bars = axis.bar(
            MODEL_NAMES,
            values,
            color=colors,
            edgecolor=INK,
            linewidth=0.7,
            width=0.65,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=3,
            fontsize=7,
            color=INK,
        )
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.14 * float(np.max(values)))
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", rotation=18)
    figure.suptitle(
        "Inverse-regression model comparison (N = 51,974)",
        fontsize=12,
        weight="semibold",
        color=INK,
        y=1.02,
    )
    figure.tight_layout(w_pad=1.6)
    paths = save_figure_formats(figure, output_base)
    plt.close(figure)
    return paths


def write_readme(
    output_dir: Path,
    comparison: FEMComparison,
) -> Path:
    """Document provenance, figure roles, notes, and draft article captions."""

    readme = output_dir / "README.md"
    text = f"""# Article-ready figures

All article figures use English labels, a white background, consistent typography, and neutral publication styling. PNG files are high-resolution raster previews; PDF and SVG files retain vector text and line work where the underlying artist is vector-based.

## `article_concept_pipeline`

- **Files:** `article_concept_pipeline.png`, `.pdf`, `.svg`
- **Type:** schematic, not a numerical field result
- **Source:** project geometry, coordinate dimensions, forward-FEM workflow, robust three-minimum extraction, and inverse-regression conventions
- **Article note:** the triangular mesh is deliberately regular and schematic. The pseudopotential panel contains exactly three marked quasi-equilibrium points. The schematic is suitable as a guide for later manual vector redrawing.
- **Caption:** *Conceptual pipeline for inverse reconstruction of RF-trap electrode displacements. A displaced four-electrode geometry is propagated through the forward FEM model, three pseudopotential minima are extracted to form supervised samples, and an inverse MLP reconstructs the eight displacement coordinates.*

## `article_fem_connections`

- **Files:** `article_fem_connections.png`, `.pdf`, `.svg`
- **Type:** real numerical output from the existing 2D FEM solver
- **Source:** row index {comparison.sample.row_index}, sample ID {comparison.sample.sample_id}, from the clean merged N=51974 dataset
- **Geometry and mesh:** the same absolute displaced geometry and the same {comparison.trap_mesh.number_of_nodes:,}-node, {comparison.trap_mesh.number_of_triangles:,}-triangle practical mesh are used in all four panels. The central mesh size is 500 µm.
- **Connections:** quadrupole checkerboard `(F1,F2,F3,F4)=(+1,-1,-1,+1) V`; in-phase `(F1,F2,F3,F4)=(+1,+1,+1,+1) V`.
- **Article note:** the displayed quantity is the project proxy `Psi = |E|^2`, not a dimensional pseudopotential energy. Red circles mark robust minima. The quadrupole panel contains {comparison.quadrupole.minima_m.shape[0]} detected minimum; the in-phase panel contains exactly {comparison.in_phase.minima_m.shape[0]} detected minima.
- **Caption:** *Real two-dimensional FEM mesh and effective-potential proxy for the same displaced electrode geometry under quadrupole checkerboard and in-phase connections. The locally refined central region is visible in the mesh panels; red markers identify robust minima of `Psi = |E|^2`.*

The associated `article_fem_connections_metadata.json` records the selected displacements, connection definitions, mesh counts, residuals, minima coordinates, and output paths.

## `article_learning_curves`

- **Files:** `article_learning_curves.png`, `.pdf`, `.svg`
- **Type:** quantitative result figure
- **Source:** `validation_results/learning_curve_merged_29995/learning_curve_metrics.csv`
- **Article note:** the three panels show the exact tracked MLP MAE, RMSE, and maximum absolute coordinate error for dataset sizes 1,000, 5,000, 10,000, 20,000, and 29,995. The maximum-error series is non-monotonic and is shown without smoothing.
- **Caption:** *Learning curves of the inverse MLP for increasing numbers of FEM-generated samples. Mean and root-mean-square coordinate errors decrease with dataset size, while the maximum coordinate error remains dominated by individual tail cases.*

## `article_model_comparison`

- **Files:** `article_model_comparison.png`, `.pdf`, `.svg`
- **Type:** quantitative result figure
- **Source:** `validation_results/inverse_model_merged_51974/metrics.csv`
- **Article note:** all bar-chart axes start at zero. The panels compare Ridge, Random Forest, and MLP on the same N=51974 train/test split.
- **Caption:** *Comparison of inverse-regression models trained on the merged N=51974 dataset. The MLP provides the lowest test MAE, RMSE, and maximum absolute coordinate error among the evaluated baseline models.*

## Reproduction

Run from the repository root:

```text
python docs/figures/generate_article_figures.py
```

The script validates the tracked metric tables before plotting and requires the selected in-phase FEM case to reproduce exactly three stored clean minima.

## Earlier draft

`inverse_reconstruction_concept.png` is an earlier Russian-language concept draft. The English `article_concept_pipeline` files are the publication-oriented replacements.
"""
    readme.write_text(text, encoding="utf-8")
    return readme


def build_all_figures(repo_root: Path) -> dict[str, object]:
    """Generate and validate every requested article figure and metadata file."""

    configure_matplotlib()
    output_dir = repo_root / "docs" / "figures"
    dataset_path = (
        repo_root
        / "validation_results"
        / "generated_dataset_merged_51974"
        / "synthetic_clean_ml.csv"
    )
    learning_metrics_path = (
        repo_root
        / "validation_results"
        / "learning_curve_merged_29995"
        / "learning_curve_metrics.csv"
    )
    model_metrics_path = (
        repo_root
        / "validation_results"
        / "inverse_model_merged_51974"
        / "metrics.csv"
    )
    validate_learning_metrics(learning_metrics_path)
    validate_model_metrics(model_metrics_path)
    sample = load_representative_sample(dataset_path, row_index=0)
    comparison = solve_connection_comparison(sample)
    concept_paths = create_concept_figure(output_dir / "article_concept_pipeline")
    fem_paths = create_fem_connections_figure(
        comparison,
        output_dir / "article_fem_connections",
    )
    metadata_path = output_dir / "article_fem_connections_metadata.json"
    write_fem_metadata(comparison, fem_paths, metadata_path)
    learning_paths = create_learning_curves_figure(
        output_dir / "article_learning_curves"
    )
    comparison_paths = create_model_comparison_figure(
        output_dir / "article_model_comparison"
    )
    readme_path = write_readme(output_dir, comparison)
    return {
        "concept_paths": [path.as_posix() for path in concept_paths],
        "fem_paths": [path.as_posix() for path in fem_paths],
        "learning_paths": [path.as_posix() for path in learning_paths],
        "model_comparison_paths": [path.as_posix() for path in comparison_paths],
        "metadata_path": metadata_path.as_posix(),
        "readme_path": readme_path.as_posix(),
        "selected_sample_id": comparison.sample.sample_id,
        "selected_row_index": comparison.sample.row_index,
        "in_phase_minimum_count": int(comparison.in_phase.minima_m.shape[0]),
        "quadrupole_minimum_count": int(
            comparison.quadrupole.minima_m.shape[0]
        ),
        "mesh_nodes": comparison.trap_mesh.number_of_nodes,
        "mesh_triangles": comparison.trap_mesh.number_of_triangles,
    }


def main() -> int:
    """Generate all requested article figures from the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    summary = build_all_figures(repo_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
