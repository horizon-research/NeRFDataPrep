import trimesh
import pyrender
import numpy as np
import argparse 
import xml.etree.ElementTree as ET
from tqdm import tqdm  # Import tqdm
import pickle
import os



def depth_and_mask_from_mesh(parsed_meta_path, mesh_path, output_folder, downsampled_factor=4.0):
    # Load the mesh
    print("Loading mesh...")
    trimesh_mesh = trimesh.load(mesh_path)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    print("Mesh loaded.")
    # Load the parsed meta file
    with open(parsed_meta_path, 'rb') as f:
        parsed_meta = pickle.load(f)
    name_poses = parsed_meta["name_poses"]
    intrinsics = parsed_meta["intrinsics"]

    scene = pyrender.Scene()
    scene.add(pyrender_mesh)

    ## change intrinsics to match the downsampled image
    intrinsics["width"] = round(intrinsics["width"]/downsampled_factor)
    intrinsics["height"] = round(intrinsics["height"]/downsampled_factor)
    intrinsics["f"] = intrinsics["f"]/downsampled_factor
    intrinsics["cx"] = intrinsics["cx"]/downsampled_factor
    intrinsics["cy"] = intrinsics["cy"]/downsampled_factor

    # Add the camera to the scene
    ## pyrender only support simple pinhole camera model, no distortion, but it work fine for the mesh rendering
    camera = pyrender.IntrinsicsCamera(fx=intrinsics["f"], fy=intrinsics["f"], cx=intrinsics["cx"], cy=intrinsics["cy"], znear=0.05, zfar=1000.0)
    camera_node = scene.add(camera)
    # Create an offscreen renderer
    renderer = pyrender.OffscreenRenderer(intrinsics["width"], intrinsics["height"])

    # use tqdm to show progress
    for name_pose in tqdm(name_poses):
        name = name_pose["name"]
        pose = name_pose["transform"]
        camera_node.matrix = pose
        # Render the scene
        color, depth = renderer.render(scene)
        np.save(output_folder+"/"+name+'_depth_fp32.npy', depth)
        mask = (depth > 0).astype(np.uint8)
        np.save(output_folder+"/"+name+'_mask_uint8.npy', mask)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth and foreground mask from mesh')
    parser.add_argument('--norm_mesh', type=str, help='Path to the cut mesh')
    parser.add_argument('--parsed_meta', type=str, help='Path to the parsed meta.pkl file')
    parser.add_argument('--downsampled_factor', type=float, default=4.0, help='Factor by which to downsample the output')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    args = parser.parse_args()
    # check if the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    depth_and_mask_from_mesh(args.parsed_meta, args.norm_mesh, args.output_folder, args.downsampled_factor)
    