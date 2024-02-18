import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import re
from bounding_box_drawer import parse_bb_from_txt
from bounding_box_drawer import update_bounding_box_lines
scene = pyrender.Scene()

def apply_transformation_to_mesh(mesh, transform_matrix):
    """
    Apply a transformation matrix to all vertices of the mesh.
    """
    # Transform all vertices using the given matrix
    transformed_vertices = np.dot(mesh.vertices, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
    mesh.vertices = transformed_vertices
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes on mesh, to help us decide the bounding box for foregorund extraction.")
    parser.add_argument("--input_mesh", help="The input mesh file.")
    parser.add_argument("--output_path", help="The output path for the cut mesh.")
    parser.add_argument("--bbox", default="garden_bbox.txt", help="The input mesh file.")
    parser.add_argument("--num_workers", default=16, help="The number of workers to use for processing the mesh faces.")
    args = parser.parse_args()
    # loading the mesh
    print("Loading mesh from", args.input_mesh) 
    mesh = trimesh.load(args.input_mesh)
    print("Mesh loaded.")

    bbox = parse_bb_from_txt(args.bbox)
    line_mesh_nodes = []
    corners, line_mesh_nodes =  update_bounding_box_lines(scene, line_mesh_nodes, bbox["cx"], bbox["cy"], bbox["cz"], bbox["rx"], bbox["ry"], bbox["rz"], bbox["lx"], bbox["ly"], bbox["lz"])
    
    # use trimesh to create a set of coordinate axes, help the bounding box adjustment (x:red y:green z:blue)
    axis_mesh = trimesh.creation.axis(origin_size=1, axis_radius=0.1, axis_length=3.0)

    # transform the trimesh axis to pyrender mesh
    axis_pyrender_mesh = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)

    # create a node and add the axis mesh to the node
    axis_node = pyrender.Node(mesh=axis_pyrender_mesh, matrix=np.eye(4))

    # add the node to the scene
    scene.add_node(axis_node)
    # construct transofmration matrix by inverse bbox
    # traslation
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = np.array([bbox["cx"], bbox["cy"], bbox["cz"]])
    transform_matrix = np.linalg.inv(transform_matrix)
    mesh = apply_transformation_to_mesh(mesh, transform_matrix)
    # rotation (rx)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(bbox["rx"]), [1, 0, 0])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    mesh = apply_transformation_to_mesh(mesh, rotation_matrix)

    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)

    ## save the mesh
    file_path = args.output_path
    mesh.export(file_path)


    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
