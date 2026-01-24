import trimesh
import os

def export_glb(meshes, out_path):
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        scene.add_geometry(m, node_name=f"building_{i}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.export(out_path)

    return out_path
