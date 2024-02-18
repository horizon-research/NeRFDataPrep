import json
import numpy as np
import argparse
import pickle
import re

import trimesh
import os 
import sys
import pyrender
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from crop_foreground.bounding_box_drawer import parse_bb_from_txt

def apply_transorm_to_mesh(mesh, transform_matrix):
    """
    Apply a transformation matrix to all vertices of the mesh.
    """
    # Transform all vertices using the given matrix
    transformed_vertices = np.dot(mesh.vertices, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
    mesh.vertices = transformed_vertices
    return mesh

def get_normalized_mesh(mesh_path, bbox):
    """
    Normalize the mesh based on the given bounding box.
    """
    # loading the mesh
    print("Loading mesh from", mesh_path)
    mesh = trimesh.load(mesh_path)
    print("Mesh loaded.")
    # translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = np.array([bbox["cx"], bbox["cy"], bbox["cz"]])
    translation_matrix = np.linalg.inv(translation_matrix)
    mesh = apply_transorm_to_mesh(mesh, translation_matrix)
    # rotation (rz)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(bbox["rz"]), [0, 0, 1])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    mesh = apply_transorm_to_mesh(mesh, rotation_matrix)
    # rotation (ry)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(bbox["ry"]), [0, 1, 0])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    mesh = apply_transorm_to_mesh(mesh, rotation_matrix)
    # rotation (rx)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(bbox["rx"]), [1, 0, 0])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    mesh = apply_transorm_to_mesh(mesh, rotation_matrix)

    # scale the mesh
    # scale_factor = max_of_lentgh 
    scale_factor = max(bbox["lx"], bbox["ly"], bbox["lz"])
    # scale the vertices
    mesh.vertices = mesh.vertices / scale_factor
    
    return mesh


def get_normalized_meta(parsed_meta, bbox):
    with open(parsed_meta, 'rb') as f:
        parsed_meta = pickle.load(f)

    bbox = parse_bb_from_txt(args.bbox)
    cx, cy, cz = bbox["cx"], bbox["cy"], bbox["cz"]
    rx, ry, rz = bbox["rx"], bbox["ry"], bbox["rz"]
    # scale_factor = max_of_lentgh 
    scale_factor = max(bbox["lx"], bbox["ly"], bbox["lz"])
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = np.array([cx, cy, cz])
    transform_matrix = np.linalg.inv(transform_matrix)

    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
    rotation_matrix_x = np.linalg.inv(rotation_matrix)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    rotation_matrix_y = np.linalg.inv(rotation_matrix)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    rotation_matrix_z = np.linalg.inv(rotation_matrix)


    name_poses = parsed_meta["name_poses"]
    for name_pose in name_poses:
        pose = name_pose["transform"]
        # apply transform 1 and 2 to pose
        pose = np.dot(transform_matrix, pose)
        pose = np.dot(rotation_matrix_z, pose)
        pose = np.dot(rotation_matrix_y, pose)
        pose = np.dot(rotation_matrix_x, pose)
        # scale the pose, only scale translation
        pose[:3, 3] = pose[:3, 3] / scale_factor
        name_pose["transform"] = pose
    
    parsed_meta["name_poses"] = name_poses

    return parsed_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the metadata file')
    parser.add_argument('--parsed_meta', type=str, help='The path to the metadata file')
    parser.add_argument("--input_mesh", help="The input mesh file.")
    parser.add_argument("--bbox", help="The input bbox file.")
    parser.add_argument('--output_mesh_path', type=str, help='The path for the normalized mesh output file')
    parser.add_argument('--output_meta_path', type=str, help='The path for the normalized poses output file')
    args = parser.parse_args()

    bbox = parse_bb_from_txt(args.bbox)
    # get the normalized mesh
    norm_mesh = get_normalized_mesh(args.input_mesh, parse_bb_from_txt(args.bbox))
    # get the normalized metadata
    norm_meta = get_normalized_meta(args.parsed_meta, bbox)


    # visualize the normalized mesh, camera poses for verification
    scene = pyrender.Scene()
    # add the axes to the scene, the normalized mesh should align with the axes
    axis_mesh = trimesh.creation.axis(origin_size=0.2, axis_radius=0.02, axis_length=1)
    axis_pyrender_mesh = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)
    axis_node = pyrender.Node(mesh=axis_pyrender_mesh, matrix=np.eye(4))
    scene.add_node(axis_node)

    # add the normalized mesh to the scene
    pyrender_norm_mesh = pyrender.Mesh.from_trimesh(norm_mesh)
    scene.add(pyrender_norm_mesh)

    # add the camera poses to the scene
    arrow_mesh = trimesh.creation.axis(origin_size=0.02, axis_radius=0.002, axis_length=0.1)
    mesh = pyrender.Mesh.from_trimesh(arrow_mesh, smooth=False)
    names_poses = norm_meta["name_poses"]
    for name_pose in names_poses:
        pose = name_pose["transform"]
        node = pyrender.Node(mesh=mesh, matrix=pose)
        scene.add_node(node)

    # save the normalized mesh
    norm_mesh.export(args.output_mesh_path)
    # save the normalized metadata
    with open(args.output_meta_path, 'wb') as f:
        pickle.dump(norm_meta, f)

    # vis
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)





