import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
def is_inside_mesh(point, bounding_box_mesh):
    """
    检查点是否在通过bounding box定义的mesh内部。
    """
    # 使用trimesh的contains方法来判断点是否在mesh内部
    return bounding_box_mesh.contains([point])[0]


def process_face(face_index, vertices, bounding_box_mesh):
    """
    检查一个面的所有顶点是否都在bounding box内，并返回面的索引如果在内部。
    """
    # 检查所有顶点是否都在bounding box内
    if all(is_inside_mesh(vertex, bounding_box_mesh) for vertex in vertices):
        return face_index
    else:
        return None

# 定义一个新的顶层函数来替换原来的lambda函数
def process_task(task, bounding_box_mesh):
    face_index, vertices = task
    return process_face(face_index, vertices, bounding_box_mesh)

def create_bounding_box_mesh(corners):
    """
    根据给定的角点创建一个bounding box mesh。
    """
    bounding_box_mesh = trimesh.convex.convex_hull(corners)
    return bounding_box_mesh


def filter_mesh_faces_inside_bounding_box(corners, trimesh_mesh, num_processes=64):
    """
    使用多进程保留在由corners定义的bounding box内的mesh面。
    """
    # 创建一个bounding box mesh
    bounding_box_mesh = create_bounding_box_mesh(corners)
    
    # 准备进程池和待处理的任务
    tasks = [(i, trimesh_mesh.vertices[face]) for i, face in tqdm(enumerate(trimesh_mesh.faces), desc="Preparing tasks")]
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 使用tqdm显示进度
        # 注意这里我们将bounding_box_mesh作为常量参数传递给所有任务
        results = list(tqdm(executor.map(process_task, tasks, [bounding_box_mesh]*len(tasks)), total=len(tasks), desc="Processing faces"))
        
    # 过滤None结果，保留内部面的索引
    inside_faces_indices = [result for result in results if result is not None]
    
    # 使用收集到的面的索引创建一个新的mesh
    inside_mesh = trimesh_mesh.submesh([inside_faces_indices], append=True)
    return inside_mesh

            
# 加载mesh
mesh_fp = 'garden/mesh.obj'  # 替换为您的mesh文件路径
trimesh_mesh = trimesh.load(mesh_fp)

# Your bounding box definition
center = np.array([-1, -0.5, -7.7])
length = np.array([9, 6, 9])  # Assuming this was meant to define the dimensions of the bounding box
min_bound = center - length / 2
max_bound = center + length / 2
scene = pyrender.Scene()
print (min_bound, max_bound)

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

# Define the angle of rotation in radians (for example, 30 degrees)
theta = np.radians(30)  # Converting degrees to radians

# Rotation matrix for X-axis
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])


corners = np.dot(corners - center, Rx.T) + center
# Define the 12 lines connecting the corners (each line is defined by two corner indices)
lines = np.array([
    # Lines along x-axis
    [0, 4], [1, 5], [2, 6], [3, 7],

    # Lines along y-axis
    [0, 3], [1, 2], [4, 7], [5, 6],

    # Lines along z-axis
    [0, 1], [3, 2], [4, 5], [7, 6]

])

# Cylinder radius for the edges
radius = 0.1

# Assuming `radius` is defined and `scene` is your pyrender.Scene object

for line in lines:
    start = corners[line[0]]
    end = corners[line[1]]
    length = np.linalg.norm(end - start)
    cyl = trimesh.creation.cylinder(radius, length, sections=8)

    # Align the cylinder with the line
    direction = end - start
    direction /= np.linalg.norm(direction)  # Normalize the direction vector
    align_mat = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cyl.apply_transform(align_mat)

    # Move it to the midpoint
    midpoint = (start + end) / 2
    translation = trimesh.transformations.translation_matrix(midpoint - cyl.centroid[:3])
    cyl.apply_transform(translation)

    mesh = pyrender.Mesh.from_trimesh(cyl)
    scene.add(mesh)

    # 使用 trimesh 创建坐标轴
    axis_mesh = trimesh.creation.axis(origin_size=1, axis_radius=0.1, axis_length=3.0)

    # 将 trimesh 的坐标轴转换为 pyrender 可用的网格（Mesh）
    axis_pyrender_mesh = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)


    # 创建一个节点，并将坐标轴网格添加到这个节点
    axis_node = pyrender.Node(mesh=axis_pyrender_mesh, matrix=np.eye(4))

    # 将节点添加到场景中
    scene.add_node(axis_node)

# 找出位于bounding box内的顶点
inside_vertices = (trimesh_mesh.vertices > min_bound) & (trimesh_mesh.vertices < max_bound)
inside_vertices = inside_vertices.all(axis=1)

# 找出所有顶点都在bounding box内的面
inside_mesh = filter_mesh_faces_inside_bounding_box(corners, trimesh_mesh)

pyrender_mesh = pyrender.Mesh.from_trimesh(inside_mesh)
scene.add(pyrender_mesh)


file_path = "cut_mesh.obj"  # 或者使用其他支持的格式，如.stl, .ply等
inside_mesh.export(file_path)


viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
