"""Conforming Gmsh mesh generation and scikit-fem conversion."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import gmsh
import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import MeshTri

from .config import MeshConfig
from .geometry import TrapGeometry


@dataclass(frozen=True)
class TrapMesh:
    """A triangular vacuum-domain mesh with classified Dirichlet nodes."""

    mesh: MeshTri
    outer_boundary_nodes: NDArray[np.int64]
    electrode_boundary_nodes: NDArray[np.int64]
    electrode_boundary_nodes_by_electrode: tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]

    @property
    def number_of_nodes(self) -> int:
        """Return the number of mesh vertices."""

        return int(self.mesh.p.shape[1])

    @property
    def number_of_triangles(self) -> int:
        """Return the number of triangular elements."""

        return int(self.mesh.t.shape[1])


@dataclass(frozen=True)
class PerforatedDiskMesh:
    """Generic disk-with-circular-holes mesh and complete boundary markers."""

    mesh: MeshTri
    outer_boundary_nodes: NDArray[np.int64]
    hole_boundary_nodes: tuple[NDArray[np.int64], ...]


def estimate_central_triangle_count(
    central_region_radius_m: float,
    central_mesh_size_m: float,
) -> int:
    """Estimate equilateral-triangle count inside a circular refined region.

    This is a preflight order-of-magnitude estimate, not an exact Gmsh count.
    It divides the disk area by ``sqrt(3) / 4 * h^2`` and deliberately excludes
    transition, electrode, and outer-domain elements.
    """

    values = np.asarray((central_region_radius_m, central_mesh_size_m), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("central radius and mesh size must be finite and positive")
    disk_area = np.pi * central_region_radius_m**2
    nominal_triangle_area = np.sqrt(3.0) * central_mesh_size_m**2 / 4.0
    return int(np.ceil(disk_area / nominal_triangle_area))


def generate_mesh(geometry: TrapGeometry, config: MeshConfig) -> TrapMesh:
    """Generate a deterministic linear triangle mesh of the perforated disk.

    Gmsh's OpenCASCADE kernel performs an exact disk-minus-four-disks Boolean
    operation.  Only first-order triangles are extracted; curved boundaries are
    therefore represented by chords whose maximum length is controlled by
    ``characteristic_length_m``.
    """

    generic = generate_perforated_disk_mesh(
        outer_radius_m=geometry.config.outer_radius_m,
        hole_centers_m=geometry.centers_m,
        hole_radii_m=(geometry.config.electrode_radius_m,) * 4,
        config=config,
    )
    if len(generic.hole_boundary_nodes) != 4:
        raise RuntimeError("trap mesh must contain exactly four electrode holes")
    electrode_nodes = np.unique(np.concatenate(generic.hole_boundary_nodes))
    return TrapMesh(
        mesh=generic.mesh,
        outer_boundary_nodes=generic.outer_boundary_nodes,
        electrode_boundary_nodes=electrode_nodes,
        electrode_boundary_nodes_by_electrode=generic.hole_boundary_nodes,
    )


def generate_perforated_disk_mesh(
    *,
    outer_radius_m: float,
    hole_centers_m: ArrayLike,
    hole_radii_m: ArrayLike,
    config: MeshConfig,
) -> PerforatedDiskMesh:
    """Mesh one circular disk minus non-overlapping circular holes with Gmsh.

    The OpenCASCADE primitives are exact circles. The returned first-order
    triangle mesh approximates them by chords controlled by the characteristic
    length, matching the production trap meshing convention.
    """

    centers = np.asarray(hole_centers_m, dtype=float)
    radii = np.asarray(hole_radii_m, dtype=float)
    if (
        not np.isfinite(outer_radius_m)
        or outer_radius_m <= 0.0
        or centers.ndim != 2
        or centers.shape[1:] != (2,)
        or radii.shape != (centers.shape[0],)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("disk and hole geometry must contain finite positive radii")
    if centers.shape[0] == 0:
        raise ValueError("at least one circular hole is required")
    if np.any(np.linalg.norm(centers, axis=1) + radii >= outer_radius_m):
        raise ValueError("every hole must lie strictly inside the outer disk")
    separation = np.linalg.norm(
        centers[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2,
    )
    pair_indices = np.triu_indices(centers.shape[0], k=1)
    radius_sums = radii[:, None] + radii[None, :]
    if np.any(separation[pair_indices] <= radius_sums[pair_indices]):
        raise ValueError("circular holes must not touch or overlap")

    owned_session = not bool(gmsh.isInitialized())
    previous_model = gmsh.model.getCurrent() if not owned_session else ""
    if owned_session:
        gmsh.initialize(argv=[], readConfigFiles=False)
    model_name = (
        "perforated_disk"
        if owned_session
        else f"perforated_disk_{uuid4().hex}"
    )
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.Algorithm", config.gmsh_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Reproducible", float(config.reproducible))
        minimum_size = config.characteristic_length_m
        maximum_size = config.characteristic_length_m
        if config.size_field is not None:
            minimum_size = min(
                config.size_field.central_mesh_size_m,
                config.size_field.electrode_boundary_mesh_size_m,
            )
            maximum_size = config.size_field.outer_mesh_size_m
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * minimum_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", maximum_size)
        gmsh.option.setNumber("Mesh.RandomFactor", config.random_factor)
        gmsh.option.setNumber("Mesh.RandomSeed", config.random_seed)
        gmsh.model.add(model_name)

        outer_tag = gmsh.model.occ.addDisk(
            0.0,
            0.0,
            0.0,
            outer_radius_m,
            outer_radius_m,
            tag=1,
        )
        hole_tags = [
            gmsh.model.occ.addDisk(
                float(center[0]),
                float(center[1]),
                0.0,
                float(radius),
                float(radius),
                tag=index + 2,
            )
            for index, (center, radius) in enumerate(zip(centers, radii, strict=True))
        ]
        surfaces, _ = gmsh.model.occ.cut(
            [(2, outer_tag)],
            [(2, tag) for tag in hole_tags],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        surface_tags = [tag for dimension, tag in surfaces if dimension == 2]
        if len(surface_tags) != 1:
            raise RuntimeError("Gmsh did not produce one connected vacuum surface")

        if config.size_field is not None:
            _configure_mesh_size_fields(
                surface_tags,
                centers,
                radii,
                config,
            )

        gmsh.model.mesh.generate(2)
        node_tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes()
        triangle_type = gmsh.model.mesh.getElementType("triangle", 1)
        _, triangle_node_tags_raw = gmsh.model.mesh.getElementsByType(triangle_type)
        points_m, triangles = _index_gmsh_mesh(
            np.asarray(node_tags_raw, dtype=np.int64),
            np.asarray(coordinates_raw, dtype=float),
            np.asarray(triangle_node_tags_raw, dtype=np.int64),
        )
    finally:
        if owned_session:
            gmsh.clear()
            gmsh.finalize()
        else:
            if gmsh.model.getCurrent() == model_name:
                gmsh.model.remove()
            if previous_model:
                gmsh.model.setCurrent(previous_model)

    mesh = MeshTri(points_m.T, triangles.T).oriented()
    outer_nodes, hole_nodes = classify_circular_boundary_nodes(
        mesh,
        outer_radius_m=outer_radius_m,
        hole_centers_m=centers,
        hole_radii_m=radii,
        tolerance_m=config.boundary_tolerance_m,
    )
    return PerforatedDiskMesh(
        mesh=mesh,
        outer_boundary_nodes=outer_nodes,
        hole_boundary_nodes=hole_nodes,
    )


def _configure_mesh_size_fields(
    surface_tags: list[int],
    hole_centers_m: NDArray[np.float64],
    hole_radii_m: NDArray[np.float64],
    config: MeshConfig,
) -> None:
    controls = config.size_field
    if controls is None:
        return
    boundary_curves = gmsh.model.getBoundary(
        [(2, tag) for tag in surface_tags],
        combined=True,
        oriented=False,
        recursive=False,
    )
    electrode_curves = _electrode_curve_tags(
        [tag for dimension, tag in boundary_curves if dimension == 1],
        hole_centers_m,
        hole_radii_m,
    )
    if not electrode_curves:
        raise RuntimeError("failed to identify electrode curves for local refinement")

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    central = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(central, "XCenter", 0.0)
    gmsh.model.mesh.field.setNumber(central, "YCenter", 0.0)
    gmsh.model.mesh.field.setNumber(central, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(central, "Radius", controls.central_region_radius_m)
    gmsh.model.mesh.field.setNumber(
        central,
        "Thickness",
        controls.central_transition_width_m,
    )
    gmsh.model.mesh.field.setNumber(central, "VIn", controls.central_mesh_size_m)
    gmsh.model.mesh.field.setNumber(central, "VOut", controls.outer_mesh_size_m)

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", electrode_curves)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
    electrode = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(electrode, "InField", distance)
    gmsh.model.mesh.field.setNumber(
        electrode,
        "SizeMin",
        controls.electrode_boundary_mesh_size_m,
    )
    gmsh.model.mesh.field.setNumber(electrode, "SizeMax", controls.outer_mesh_size_m)
    gmsh.model.mesh.field.setNumber(electrode, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(
        electrode,
        "DistMax",
        controls.electrode_transition_width_m,
    )

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", [central, electrode])
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)


def _electrode_curve_tags(
    curve_tags: list[int],
    centers_m: NDArray[np.float64],
    radii_m: NDArray[np.float64],
) -> list[int]:
    tags = []
    # OpenCASCADE expands curve bounding boxes by its modelling tolerance.
    # One micrometre is still four orders of magnitude smaller than the real
    # electrode radius and reliably distinguishes every trap boundary.
    tolerance = 1.0e-6
    for tag in curve_tags:
        minimum_x, minimum_y, _, maximum_x, maximum_y, _ = gmsh.model.getBoundingBox(
            1,
            tag,
        )
        center = np.asarray(
            ((minimum_x + maximum_x) / 2.0, (minimum_y + maximum_y) / 2.0)
        )
        radius = 0.25 * ((maximum_x - minimum_x) + (maximum_y - minimum_y))
        if any(
            np.linalg.norm(center - expected_center) <= tolerance
            and abs(radius - expected_radius) <= tolerance
            for expected_center, expected_radius in zip(
                centers_m,
                radii_m,
                strict=True,
            )
        ):
            tags.append(tag)
    return tags


def _index_gmsh_mesh(
    node_tags: NDArray[np.int64],
    coordinates: NDArray[np.float64],
    triangle_node_tags: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if node_tags.size == 0 or triangle_node_tags.size == 0:
        raise RuntimeError("Gmsh returned an empty two-dimensional mesh")
    points_m = coordinates.reshape(-1, 3)[:, :2]
    tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
    try:
        flat_triangles = np.fromiter(
            (tag_to_index[int(tag)] for tag in triangle_node_tags),
            dtype=np.int64,
            count=triangle_node_tags.size,
        )
    except KeyError as error:
        raise RuntimeError("a Gmsh triangle references an unknown node") from error
    return points_m, flat_triangles.reshape(-1, 3)


def classify_circular_boundary_nodes(
    mesh: MeshTri,
    *,
    outer_radius_m: float,
    hole_centers_m: ArrayLike,
    hole_radii_m: ArrayLike,
    tolerance_m: float,
) -> tuple[
    NDArray[np.int64],
    tuple[NDArray[np.int64], ...],
]:
    """Classify and validate every boundary vertex of a perforated disk."""

    points = mesh.p.T
    centers = np.asarray(hole_centers_m, dtype=float)
    radii = np.asarray(hole_radii_m, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (2,) or radii.shape != (len(centers),):
        raise ValueError("hole centers and radii have inconsistent shapes")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be finite and positive")
    outer_mask = np.isclose(
        np.linalg.norm(points, axis=1),
        outer_radius_m,
        rtol=0.0,
        atol=tolerance_m,
    )
    hole_masks = []
    for center, radius in zip(centers, radii, strict=True):
        hole_masks.append(
            np.isclose(
                np.linalg.norm(points - center, axis=1),
                radius,
                rtol=0.0,
                atol=tolerance_m,
            )
        )
    outer_nodes = np.flatnonzero(outer_mask).astype(np.int64)
    nodes_by_hole = tuple(
        np.flatnonzero(mask).astype(np.int64) for mask in hole_masks
    )
    if outer_nodes.size == 0 or any(nodes.size == 0 for nodes in nodes_by_hole):
        raise RuntimeError("failed to classify all Dirichlet boundaries")
    hole_nodes = np.unique(np.concatenate(nodes_by_hole))
    if hole_nodes.size != sum(nodes.size for nodes in nodes_by_hole):
        raise RuntimeError("circular-hole boundary classifications overlap")
    if np.intersect1d(outer_nodes, hole_nodes).size:
        raise RuntimeError("outer and hole boundary classifications overlap")
    classified = np.union1d(outer_nodes, hole_nodes)
    unclassified = np.setdiff1d(mesh.boundary_nodes(), classified)
    if unclassified.size:
        raise RuntimeError(
            "some mesh boundary nodes were not classified; increase boundary_tolerance_m"
        )
    return outer_nodes, nodes_by_hole


def nearest_mesh_facet(
    mesh: MeshTri,
    position_m: ArrayLike,
) -> tuple[float, int]:
    """Return Euclidean distance and index of the nearest mesh facet."""

    position = np.asarray(position_m, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("position_m must be one finite two-dimensional point")
    facets = mesh.facets.T
    points = mesh.p.T
    starts = points[facets[:, 0]]
    ends = points[facets[:, 1]]
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.clip(
        np.einsum("ij,ij->i", position - starts, vectors) / lengths_squared,
        0.0,
        1.0,
    )
    projections = starts + fractions[:, np.newaxis] * vectors
    distances = np.linalg.norm(position - projections, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def nearest_internal_mesh_facet(
    mesh: MeshTri,
    position_m: ArrayLike,
) -> tuple[float, int]:
    """Return distance and index of the nearest two-sided mesh facet."""

    position = np.asarray(position_m, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("position_m must be one finite two-dimensional point")
    internal_indices = np.flatnonzero(np.all(mesh.f2t >= 0, axis=0))
    if internal_indices.size == 0:
        raise ValueError("mesh contains no internal facets")
    facets = mesh.facets[:, internal_indices].T
    points = mesh.p.T
    starts = points[facets[:, 0]]
    ends = points[facets[:, 1]]
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.clip(
        np.einsum("ij,ij->i", position - starts, vectors) / lengths_squared,
        0.0,
        1.0,
    )
    projections = starts + fractions[:, np.newaxis] * vectors
    distances = np.linalg.norm(position - projections, axis=1)
    local_index = int(np.argmin(distances))
    return float(distances[local_index]), int(internal_indices[local_index])
