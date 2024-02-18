
import json
import numpy as np
import argparse
import pickle
import re

import trimesh

rotation_matrix_180_y = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

rotation_matrix_180_z = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the metadata file')
    parser.add_argument('--in_meta', type=str, help='The path to the metadata file')
    parser.add_argument('--output_path', type=str, help='The path to the fixed output file')
    args = parser.parse_args()
    with open(args.in_meta, 'rb') as f:
        parsed_meta = pickle.load(f)

    name_poses = parsed_meta["name_poses"]
    fixation_matrix = rotation_matrix_180_y @ rotation_matrix_180_z
    for name_pose in name_poses:
        pose = name_pose["transform"]
        pose = np.dot(pose, fixation_matrix)
        name_pose["transform"] = pose
    
    parsed_meta["name_poses"] = name_poses

    with open(args.output_path, 'wb') as file:
        pickle.dump(parsed_meta, file)