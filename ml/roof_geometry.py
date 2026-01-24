# ml/roof_geometry.py

import numpy as np
import trimesh
from shapely.geometry import Polygon


# --------------------------------------------------
# ROOF TYPE INFERENCE
# --------------------------------------------------
def infer_roof_type(polygon):
    """
    Simple heuristic roof classifier.
    LOD 2.5 baseline.
    """

    area = polygon.area
    coords = list(polygon.exterior.coords)
    edges = len(coords) - 1

    # Small buildings → flat
    if area < 150:
        return "flat"

    # Rectangular-ish → gabled
    if edges == 4:
        return "gabled"

    # Everything else → barrel / curved proxy
    return "barrel"

# def infer_roof_type(footprint: Polygon) -> str:
#     """
#     Simple heuristic placeholder.
#     Later: shadow / OSM / ML.
#     """
#     area = footprint.area
#     if area > 300:
#         return "gabled"
#     return "flat"


# --------------------------------------------------
# FLAT ROOF (LOD1)
# --------------------------------------------------

def flat_roof(poly, height, base=0.0):
    """
    Create an extruded LOD1 building mesh from a footprint.
    """

    if height <= 0:
        height = 3.0  # safety

    # Exterior coords (drop closing point)
    coords = list(poly.exterior.coords)[:-1]

    # Base and top vertices
    base_vertices = [(x, y, base) for x, y in coords]
    top_vertices  = [(x, y, base + height) for x, y in coords]

    vertices = base_vertices + top_vertices

    n = len(coords)
    faces = []

    # Side faces
    for i in range(n):
        j = (i + 1) % n

        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    # Roof face (fan triangulation)
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        process=True
    )

    return mesh


# --------------------------------------------------
# GABLED ROOF (LOD2)
# --------------------------------------------------

def gabled_roof(
    footprint: Polygon,
    height: float,
    roof_ratio: float = 0.35
) -> trimesh.Trimesh:

    wall_height = height * (1 - roof_ratio)

    walls = trimesh.creation.extrude_polygon(
        footprint,
        wall_height,
        engine="earcut"
    )

    verts_2d, faces = trimesh.creation.triangulate_polygon(
        footprint,
        engine="earcut"
    )

    center = verts_2d.mean(axis=0)
    max_dist = np.max(np.linalg.norm(verts_2d - center, axis=1)) + 1e-6

    roof_vertices = []
    for v in verts_2d:
        d = np.linalg.norm(v - center)
        z = wall_height + roof_ratio * height * (1 - d / max_dist)
        roof_vertices.append([v[0], v[1], z])

    roof = trimesh.Trimesh(
        vertices=np.array(roof_vertices),
        faces=faces,
        process=True
    )

    mesh = trimesh.util.concatenate([walls, roof])
    mesh.merge_vertices()
    return mesh


# --------------------------------------------------
# BARREL (CURVED) ROOF (LOD2.5)
# --------------------------------------------------

def barrel_roof(
    footprint: Polygon,
    height: float,
    roof_ratio: float = 0.4,
    segments: int = 12
) -> trimesh.Trimesh:
    """
    Simple barrel vault roof.
    """

    wall_height = height * (1 - roof_ratio)
    roof_height = height * roof_ratio

    # ---- Walls ----
    walls = trimesh.creation.extrude_polygon(
        footprint,
        wall_height,
        engine="earcut"
    )

    # ---- Footprint bounds ----
    minx, miny, maxx, maxy = footprint.bounds
    length = maxx - minx
    width = maxy - miny

    # Choose barrel axis
    axis_x = length >= width

    # ---- Sample curve ----
    t = np.linspace(-1, 1, segments)
    curve = roof_height * np.sqrt(1 - t**2)

    verts_2d, faces = trimesh.creation.triangulate_polygon(
        footprint,
        engine="earcut"
    )

    roof_vertices = []
    for v in verts_2d:
        if axis_x:
            u = (v[1] - miny) / width * 2 - 1
        else:
            u = (v[0] - minx) / length * 2 - 1

        idx = np.clip(
            int((u + 1) / 2 * (segments - 1)),
            0,
            segments - 1
        )

        z = wall_height + curve[idx]
        roof_vertices.append([v[0], v[1], z])

    roof = trimesh.Trimesh(
        vertices=np.array(roof_vertices),
        faces=faces,
        process=True
    )

    mesh = trimesh.util.concatenate([walls, roof])
    mesh.merge_vertices()
    return mesh
