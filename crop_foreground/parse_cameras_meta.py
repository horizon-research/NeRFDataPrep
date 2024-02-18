import trimesh
import pyrender
import numpy as np
import argparse 
import xml.etree.ElementTree as ET
from tqdm import tqdm  # Import tqdm
import pickle


def string_to_transform_matrix(transform_str):
    """
    Convert a string representation of a 4x4 transformation matrix
    into a NumPy array.

    Parameters:
    transform_str (str): A string containing 16 floating-point numbers separated by spaces.

    Returns:
    np.ndarray: A 4x4 NumPy array representing the transformation matrix.
    """
    # Split the string into a list of strings, then convert each to float
    transform_values = [float(value) for value in transform_str.split()]
    
    # Convert the list of floats into a 4x4 NumPy array
    transform_matrix = np.array(transform_values).reshape(4, 4)
    
    return transform_matrix

def parse_intrinsics(tree):
    result = dict()
    root = tree.getroot()
    # Navigate to the calibration element of the second sensor (id="1")
    # This assumes that the structure is consistent with your provided XML snippet
    i = 0
    while True:
        sensor = root.find(".//sensor[@id='{}']".format(i))
        if sensor.find('calibration'):
            break
        i += 1
        if i > 10:
            raise ValueError("No calibration found")
    calibration = sensor.find('calibration')


    # Extract the parameters
    resolution = calibration.find('resolution')
    result["width"] = float(resolution.get('width'))
    result["height"] = float(resolution.get('height'))

    result["f"] = float(calibration.find('f').text)
    result["cx"] = float(calibration.find('cx').text)
    result["cy"] = float(calibration.find('cy').text)
    result["k1"] = float(calibration.find('k1').text)
    result["k2"] = float(calibration.find('k2').text)
    result["k3"] = float(calibration.find('k3').text)
    result["p1"] = float(calibration.find('p1').text)
    result["p2"] = float(calibration.find('p2').text)

    return result

def parse_name_pose(tree):
    root = tree.getroot()
    cameras = root.findall(".//camera")
    results = list()
    for camera in cameras:
        name = camera.get('label')
        transform = camera.find('transform').text if camera.find('transform') is not None else 'N/A'
        if transform == 'N/A':
            print(f"Transform not found for camera {name}")
            continue
        result = dict()
        result["name"] = name
        result["transform"] = string_to_transform_matrix(transform)
        results.append(result)

    return results

def parse_meta(meta_xml_fp, output_path):
    tree = ET.parse(meta_xml_fp)
    intrinsics = parse_intrinsics(tree)
    intrinsics["cx"] = intrinsics["cx"] + intrinsics["width"] / 2
    intrinsics["cy"] = intrinsics["cy"] + intrinsics["height"] / 2
    name_poses = parse_name_pose(tree)

    save_result = dict()
    # use pickle to save intrinsic and extrinsic parameters
    save_result["intrinsics"] = intrinsics
    save_result["name_poses"] = name_poses


    with open(output_path, 'wb') as file:
        pickle.dump(save_result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the meta.xml file to get the camera poses and intrinsics')
    parser.add_argument('--meta_file', type=str, help='Path to the meta.xml file')
    parser.add_argument('--output_path', type=str, help='Path to store the parse result')
    args = parser.parse_args()
    parse_meta(args.meta_file, args.output_path)