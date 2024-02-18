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


def is_inside_mesh(point, bounding_box_mesh):
    """
    check if a point is inside a mesh defined by a bounding box.
    """
    # 使用trimesh的contains方法来判断点是否在mesh内部
    return bounding_box_mesh.contains([point])[0]


def process_face(face_index, vertices, bounding_box_mesh):
    """
    check if all vertices of a face are inside the bounding box and return the face index if they are.
    """
    if all(is_inside_mesh(vertex, bounding_box_mesh) for vertex in vertices):
        return face_index
    else:
        return None

def process_task(task):
    face_index, vertices, bounding_box_mesh = task
    return process_face(face_index, vertices, bounding_box_mesh)

def create_bounding_box_mesh(corners):
    """
    create a bounding box mesh based on the given corners.
    """
    bounding_box_mesh = trimesh.convex.convex_hull(corners)
    return bounding_box_mesh


def filter_mesh_faces_inside_bounding_box(corners, trimesh_mesh, num_processes):
    """
    use multiple processes to retain the mesh faces inside the bounding box defined by corners.
    """
    # create a bounding box mesh
    bounding_box_mesh = create_bounding_box_mesh(corners)
    
    # prepare the process pool and the tasks to be processed
    tasks = [(i, trimesh_mesh.vertices[face], bounding_box_mesh) for i, face in tqdm(enumerate(trimesh_mesh.faces), total=len(trimesh_mesh.faces), desc="Preparing tasks")]
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Processing faces"))
        
    # filter out the None results and retain the indices of the inside faces
    inside_faces_indices = [result for result in results if result is not None]
    
    # create a new mesh using the collected indices of the faces
    inside_mesh = trimesh_mesh.submesh([inside_faces_indices], append=True)
    return inside_mesh



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
