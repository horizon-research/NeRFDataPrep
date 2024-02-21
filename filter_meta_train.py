
import json
import numpy as np
import argparse
import pickle
import re

import trimesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the metadata file')
    parser.add_argument('--in_meta', type=str, help='The path to the metadata file')
    parser.add_argument('--output_path', type=str, help='The path to the fixed output file')
    args = parser.parse_args()
    with open(args.in_meta, 'rb') as f:
        parsed_meta = pickle.load(f)

    name_poses = parsed_meta["name_poses"]
    new_name_poses = []
    for idx, name_pose in enumerate(name_poses):
        if idx >= 900 or idx % 20 == 0:
            continue
        else:
            new_name_poses.append(name_pose)
    
    parsed_meta["name_poses"] = new_name_poses

    with open(args.output_path, 'wb') as file:
        pickle.dump(parsed_meta, file)