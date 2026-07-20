"""Conforming Gmsh mesh generation and scikit-fem conversion."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import gmsh
import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri

from .config import MeshConfig
from .geometry import TrapGeometry


@dataclass(frozen=True)
class TrapMesh:
    """A triangular vacuum-domain mesh with classified Dirichlet nodes."""

    mesh: MeshTri
    outer_boundary_nodes: NDArray[np.int64]
    electrode_boundary_nodes: NDArray[np.int64]

    @property
    def number_of_nodes(self) -> int:
        """Return the number of mesh vertices."""

        return int(self.mesh.p.shape[1])

    @property
    def number_of_triangles(self) -> int:
        """Return the number of triangular elements."""

        return int(self.mesh.t.shape[1])


def generate_mesh(geometry: TrapGeometry, config: MeshConfig) -> TrapMesh:
    """Generate a deterministic linear triangle mesh of the perforated disk.

    Gmsh's OpenCASCADE kernel performs an exact disk-minus-four-disks Boolean
    operation.  Only first-order triangles are extracted; curved boundaries are
    therefore represented by chords whose maximum length is controlled by
    ``characteristic_length_m``.
    """

    owned_session = not bool(gmsh.isInitialized())
    previous_model = gmsh.model.getCurrent() if not owned_session else ""
    if owned_session:
        gmsh.initialize(argv=[], readConfigFiles=False)
    model_name = "rf_trap" if owned_session else f"rf_trap_{uuid4().hex}"
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.Algorithm", config.gmsh_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Reproducible", float(config.reproducible))
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * config.characteristic_length_m)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.characteristic_length_m)
        gmsh.option.setNumber("Mesh.RandomFactor", config.random_factor)
        gmsh.option.setNumber("Mesh.RandomSeed", config.random_seed)
        gmsh.model.add(model_name)

        outer_tag = gmsh.model.occ.addDisk(
            0.0,
            0.0,
            0.0,
            geometry.config.outer_radius_m,
            geometry.config.outer_radius_m,
            tag=1,
        )
        hole_tags = [
            gmsh.model.occ.addDisk(
                float(center[0]),
                float(center[1]),
                0.0,
                geometry.config.electrode_radius_m,
                geometry.config.electrode_radius_m,
                tag=index + 2,
            )
            for index, center in enumerate(geometry.centers_m)
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
    outer_nodes, electrode_nodes = _classify_boundary_nodes(mesh, geometry, config)
    return TrapMesh(
        mesh=mesh,
        outer_boundary_nodes=outer_nodes,
        electrode_boundary_nodes=electrode_nodes,
    )


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


def _classify_boundary_nodes(
    mesh: MeshTri,
    geometry: TrapGeometry,
    config: MeshConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    points = mesh.p.T
    tolerance = config.boundary_tolerance_m
    outer_mask = np.isclose(
        np.linalg.norm(points, axis=1),
        geometry.config.outer_radius_m,
        rtol=0.0,
        atol=tolerance,
    )
    electrode_mask = np.zeros(points.shape[0], dtype=bool)
    for center in geometry.centers_m:
        electrode_mask |= np.isclose(
            np.linalg.norm(points - center, axis=1),
            geometry.config.electrode_radius_m,
            rtol=0.0,
            atol=tolerance,
        )
    outer_nodes = np.flatnonzero(outer_mask).astype(np.int64)
    electrode_nodes = np.flatnonzero(electrode_mask).astype(np.int64)
    if outer_nodes.size == 0 or electrode_nodes.size == 0:
        raise RuntimeError("failed to classify all Dirichlet boundaries")
    if np.intersect1d(outer_nodes, electrode_nodes).size:
        raise RuntimeError("outer and electrode boundary classifications overlap")
    classified = np.union1d(outer_nodes, electrode_nodes)
    unclassified = np.setdiff1d(mesh.boundary_nodes(), classified)
    if unclassified.size:
        raise RuntimeError(
            "some mesh boundary nodes were not classified; increase boundary_tolerance_m"
        )
    return outer_nodes, electrode_nodes
