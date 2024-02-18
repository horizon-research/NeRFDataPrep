import argparse
import trimesh
import pyrender
import numpy as np
import re
import pickle

def update_bounding_box_lines(scene, line_mesh_nodes, cx, cy, cz, rx, ry, rz, lx, ly, lz):
    # define the bounding box
    center = np.array([cx, cy, cz])
    length = np.array([lx, ly, lz])
    min_bound = center - length / 2
    max_bound = center + length / 2
    x_theta = np.radians(rx)
    y_theta = np.radians(ry)
    z_theta = np.radians(rz)

    # Rotation matrix for X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_theta), -np.sin(x_theta)],
        [0, np.sin(x_theta), np.cos(x_theta)]
    ])

    # Rotation matrix for Y-axis
    Ry = np.array([
        [np.cos(y_theta), 0, np.sin(y_theta)],
        [0, 1, 0],
        [-np.sin(y_theta), 0, np.cos(y_theta)]
    ])

    # Rotation matrix for Z-axis
    Rz = np.array([
        [np.cos(z_theta), -np.sin(z_theta), 0],
        [np.sin(z_theta), np.cos(z_theta), 0],
        [0, 0, 1]
    ])

    # Calculate the 8 corners of the bounding box
    corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
    ])

    corners = np.dot(corners - center, Rx.T) + center
    corners = np.dot(corners - center, Ry.T) + center
    corners = np.dot(corners - center, Rz.T) + center

    # Define the 12 lines connecting the corners (each line is defined by two corner indices)
    lines = np.array([
        # Lines along x-axis
        [0, 4], [1, 5], [2, 6], [3, 7],

        # Lines along y-axis
        [0, 3], [1, 2], [4, 7], [5, 6],

        # Lines along z-axis
        [0, 1], [3, 2], [4, 5], [7, 6]

    ])

    # First, remove existing line meshes from the scene
    for node in line_mesh_nodes:
        scene.remove_node(node)
    line_mesh_nodes = []  # Clear the list

    # Then, add new line meshes based on updated corners
    radius = 0.1
    for line in lines:
        start = corners[line[0]]
        end = corners[line[1]]
        length = np.linalg.norm(end - start)
        cyl = trimesh.creation.cylinder(radius, length, sections=8)

        # Align, move, and create the mesh
        direction = (end - start) / length
        align_mat = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cyl.apply_transform(align_mat)
        midpoint = (start + end) / 2
        translation = trimesh.transformations.translation_matrix(midpoint - cyl.centroid[:3])
        cyl.apply_transform(translation)

        mesh = pyrender.Mesh.from_trimesh(cyl)
        node = scene.add(mesh, pose=np.eye(4))
        line_mesh_nodes.append(node)  # Keep track of the node

    return corners, line_mesh_nodes

def parse_bb_from_txt(fname='garden_bbox.txt'):
    # Define the pattern to match variables like "cx = -1;"
    # This pattern matches:
    # - variable names composed of letters (and underscores)
    # - an optional space around the equals sign
    # - a negative or positive number (integer or decimal)
    pattern = re.compile(r'([a-z]+)\s*=\s*(-?\d+\.?\d*);', re.IGNORECASE)

    # Dictionary to hold the parsed values
    values = {}

    # Open and read the text file
    with open(fname, 'r') as file:
        content = file.read()
        
        # Find all matches of the pattern
        matches = pattern.findall(content)
        
        # Iterate over matches and store them in the dictionary
        for var_name, var_value in matches:
            values[var_name] = float(var_value)  # Convert the string to a float

    # Print the parsed values to verify
    for var_name, var_value in values.items():
        print(f'{var_name} = {var_value}')

    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes on mesh, to help us decide the bounding box for foregorund extraction.")
    parser.add_argument("--input_mesh", help="The input mesh file.")
    parser.add_argument("--bbox", default="garden_bbox.txt", help="The input mesh file.")
    parser.add_argument("--parsed_meta",  help="parsed_meta")

    args = parser.parse_args()

    # Create a pyrender scene
    scene = pyrender.Scene()

    line_mesh_nodes = []  # Keep track of the line mesh nodes

    # loading the mesh
    print("Loading mesh from", args.input_mesh) 
    mesh = trimesh.load(args.input_mesh)
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)
    print("Mesh loaded.")

    # use trimesh to create a set of coordinate axes, help the bounding box adjustment (x:red y:green z:blue)
    axis_mesh = trimesh.creation.axis(origin_size=1, axis_radius=0.1, axis_length=3.0)

    # transform the trimesh axis to pyrender mesh
    axis_pyrender_mesh = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)

    # create a node and add the axis mesh to the node
    axis_node = pyrender.Node(mesh=axis_pyrender_mesh, matrix=np.eye(4))

    # add the node to the scene
    scene.add_node(axis_node)



    # add the camera poses to the scene
    arrow_mesh = trimesh.creation.axis(origin_size=0.2, axis_radius=0.02, axis_length=1)
    mesh = pyrender.Mesh.from_trimesh(arrow_mesh, smooth=False)
    with open(args.parsed_meta, "rb") as f:
        parsed_meta = pickle.load(f)
    names_poses = parsed_meta["name_poses"]
    for name_pose in names_poses:
        pose = name_pose["transform"]
        node = pyrender.Node(mesh=mesh, matrix=pose)
        scene.add_node(node)


    # parse the bounding box from the text file
    bbox = parse_bb_from_txt(args.bbox)
    cx = bbox['cx']
    cy = bbox['cy']
    cz = bbox['cz']
    rx = bbox['rx']
    ry = bbox['ry']
    rz = bbox['rz']
    lx = bbox['lx']
    ly = bbox['ly']
    lz = bbox['lz']
    while True:
        corners, line_mesh_nodes = update_bounding_box_lines(scene, line_mesh_nodes, cx, cy, cz, rx, ry, rz, lx, ly, lz)
        # tell the user type q to quit or modified bbox then press enter
        print("If the bounding box is fine, close pyviewer, type q+enter to quit.")
        print("Otherwise, please modify the bounding box in the bbox.txt file, close pyviewer, and press enter. The program will reparse it and update the bounding box.")
        # render the scene
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
        user_input = input()
        if user_input == "q":
            break
        else:
            bbox = parse_bb_from_txt(args.bbox)
            cx = bbox['cx']
            cy = bbox['cy']
            cz = bbox['cz']
            rx = bbox['rx']
            ry = bbox['ry']
            rz = bbox['rz']
            lx = bbox['lx']
            ly = bbox['ly']
            lz = bbox['lz']



