import numpy as np 
import cv2
import math
import open3d as o3d
import copy
import json
from skimage.metrics import structural_similarity as ssim
from skvideo.io import FFmpegWriter

from tqdm import tqdm
import pickle
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
# def PSNR(original, compressed, depth_map):
#     mse = np.mean((original[depth_map>0] - compressed[depth_map>0]) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
#     return psnr

def PSNR(original, compressed, depth_map):
    # import ipdb; ipdb.set_trace()
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# camera coord direction is 
#                   ^
#  z out to screen  |
#  y to up          |
#  x to right       *---->
#
# therefore, we have to convert y to negative
def compute_xy(depth_map, fl_x, fl_y, cx, cy):

    img_height, img_width, _ = depth_map.shape

    # 0.5 is half pixel offset
    x_coord = np.arange(0, img_width, dtype=float) + 0.5 # (img_width,)
    x_coord = np.expand_dims(x_coord, axis=0) # expand to (1, img_width)
    x_coord = np.repeat(x_coord, img_height, axis=0) # repeat in y-axis `img_height` times
    
    y_coord = np.arange(0, img_height, dtype=float) + 0.5 # (img_height,)
    y_coord = np.expand_dims(y_coord, axis=1) # expand to (img_height, 1)
    y_coord = np.repeat(y_coord, img_width, axis=1) # repeat in x-axis `img_width` times

    # X =(Depth*dx)/fl
    x_val = (x_coord - cx) * depth_map[:, :, 0] / fl_x # np.sqrt(fl_x**2 + (x_coord - cx)**2)
    y_val = -(y_coord - cy) * depth_map[:, :, 0] / fl_y # np.sqrt(fl_y**2 + (y_coord - cy)**2)

    return x_val, y_val

def compose_pcd(depth_map, img, fl_x, fl_y, cx, cy):
    img_height, img_width, _ = depth_map.shape
    # first calculate the x and y values in camera coordinate system
    x_val, y_val = compute_xy(depth_map, fl_x, fl_y, cx, cy)
    # combine into a 3d point cloud
    # camera coord direction is 
    #                   ^
    #  z out to screen  |
    #  y to up          |
    #  x to right       *---->
    #
    # therefore, we have to convert y and z to negative, here y is already negated.
    points = np.stack((x_val, y_val, -depth_map[:, :, 0]), axis=2)
    points = np.reshape(points, (img_height * img_width, 3))
    colors = np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img_height * img_width, 3))/255.

    # construct point cloud data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def pc_to_rgb(pcd, fl_x, fl_y, cx, cy, img_width, img_height):
    restored_img = np.zeros((img_height, img_width, 3))
    z_buffer = np.zeros((img_height, img_width))
    z_buffer[:, :] = 10
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    for i, pt in enumerate(points):
        rgb = colors[i]

        if np.sum(rgb) == 0:
            continue

        x, y, z = pt
        z = np.abs(z) # needs to negate z value
        x_coord = int(cx + x / z * fl_x - 0.5)
        y_coord = int(cy - y / z * fl_y - 0.5) # - 400
        # print(i, y_coord, x_coord, x, y, z)

        if y_coord < 0 or y_coord >= img_height or x_coord < 0 or x_coord >= img_width:
            continue

        if z < z_buffer[y_coord, x_coord]:
            z_buffer[y_coord, x_coord] = z
            restored_img[y_coord, x_coord] = rgb * 256

    restored_img = np.clip(restored_img, 0, 255)

    return cv2.cvtColor(restored_img.astype(np.uint8), cv2.COLOR_BGR2RGB), z_buffer

def fill_disocclusion(restored_img, ref_img, ref_depth, z_buffer):
    fill_pixel_cnt = 0.
    img_height, img_width = ref_depth.shape

    for r in range(img_height):
        for c in range(img_width):
            if ref_depth[r, c] > 0 and ref_depth[r, c] < z_buffer[r, c]:
                restored_img[r, c] = ref_img[r, c]
                fill_pixel_cnt += 1
            if ref_depth[r, c] == 0:
                restored_img[r, c] = ref_img[r, c]

    return restored_img, fill_pixel_cnt/img_width/img_height 

def load_transform_file(fn):

    data = json.load(open(fn))

    transform_list = []
    for i in range(len(data["frames"])):
        transform_list.append(
            np.array(data["frames"][i]["transform_matrix"])
        )

    return transform_list


def inverse_transform_mat(mat):
    rot_mat = mat[:3, :3]
    inverse_rot = np.linalg.inv(rot_mat)
    inverse_translation = np.dot(inverse_rot, mat[:3, 3])
    inverse_mat = np.zeros((4, 4))

    inverse_mat[:3, :3] = inverse_rot
    inverse_mat[:3, 3] = -inverse_translation

    inverse_mat[3, 3] = 1

    return inverse_mat

def visualize_pcd(pcd_list, img_width, img_height):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

fixation_matrix = rotation_matrix_180_y @ rotation_matrix_180_z

def main():
    scene_folder = "./garden"
    in_ex_param_fp = scene_folder + "/in_ex_param.pkl"
    with open(in_ex_param_fp, "rb") as f:
        in_ex_param = pickle.load(f)
    
    fl_x = in_ex_param["intrinsics"]["f"]
    fl_y = in_ex_param["intrinsics"]["f"]
    cx = in_ex_param["intrinsics"]["cx"]
    cy = in_ex_param["intrinsics"]["cy"]
    img_width = int(in_ex_param["intrinsics"]["width"])
    img_height = int(in_ex_param["intrinsics"]["height"])
    name_poses = in_ex_param["name_poses"]

    # downsampe 4x
    img_width = img_width // 4
    img_height = img_height // 4
    fl_x = fl_x / 4
    fl_y = fl_y / 4
    cx = cx / 4
    cy = cy / 4

    item_name = "garden"

    prev_pcd = None

    total_exp_psnr = []
    total_act_psnr = []
    total_resize_x2_psnr = []
    total_resize_x4_psnr = []
    total_ssim = []
    total_fill_pct = []

    # tentatively write a demo video
    writer = FFmpegWriter(
        "./%s_demo.mp4" % item_name,
        inputdict={
            '-r': str(3),
        },
        outputdict={
            '-r': str(3),
        }
    )

    for i, name_pose in tqdm(enumerate(name_poses)):
        name = name_pose["name"]
        pose = name_pose["transform"]
        depth_fn = "%s/%s_depth_fp32.npy" % (scene_folder, name)
        ref_fn = "%s/%s.JPG" % (scene_folder, name)
        act_fn = "%s/%s.JPG" % (scene_folder, name)
        

        trans_mat = pose @ fixation_matrix
        inverse_trans_mat = inverse_transform_mat(trans_mat)
        depth_map = np.load(depth_fn)

        act_img = cv2.imread(act_fn)
        ref_img = cv2.imread(ref_fn)

        # downsample 4x
        act_img = cv2.resize(act_img, (img_width, img_height))
        ref_img = cv2.resize(ref_img, (img_width, img_height))
        depth_map = cv2.resize(depth_map, (img_width, img_height))
        depth_map = depth_map[:, :, np.newaxis]

        depth_map[depth_map < 0.1] = 0
        depth_map[depth_map > 9.9] = 0

        
        # print(np.max(depth_map), depth_map.shape)
        # cv2.imshow("depth", depth_map/np.max(depth_map))
        # cv2.waitKey(0)

        # conpose point cloud
        pcd = compose_pcd(depth_map, act_img, fl_x, fl_y, cx, cy)
        # transform point cloud from camera coordinate to world coordinate
        # so that all point clouds will be at the same coordinate system
        pcd = pcd.transform(trans_mat)

        # apply transformation matrix
        if prev_pcd != None:
            # move back to its own camera coordinate for image capturing
            # visualize_pcd(
            #     [
            #         copy.deepcopy(prev_pcd).transform(inverse_trans_mat), # previous transformed pcd
            #         copy.deepcopy(pcd).transform(inverse_trans_mat)       # current pcd
            #     ],
            #     img_width,
            #     img_height
            # )

            restored_img, z_buffer = pc_to_rgb(
                copy.deepcopy(prev_pcd).transform(inverse_trans_mat), 
                fl_x, fl_y, cx, cy, 
                img_width, img_height
            )
            restored_img, fill_pct = fill_disocclusion(restored_img, act_img, depth_map[:, :, 0], z_buffer)
            diff = np.abs(restored_img.astype(np.float32) - ref_img.astype(np.float32)).astype(np.uint8)
            comb = np.hstack((restored_img, ref_img, diff))
            cv2.imshow("restored", comb)
            writer.writeFrame(
                cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
            )

            # save the restored image
            cv2.imwrite(
                "./restored.png" ,
                comb
            )

            # import ipdb; ipdb.set_trace()
            cv2.waitKey(10)


            exp_psnr_val = PSNR(restored_img, ref_img, depth_map)
            act_psnr_val = PSNR(act_img, ref_img, depth_map)
            # import ipdb; ipdb.set_trace()
            resize_x2_psnr_val = PSNR(
                cv2.resize(act_img[::2, ::2, :], (img_width, img_height)), 
                ref_img, depth_map
            )
            resize_x4_psnr_val = PSNR(
                cv2.resize(act_img[::4, ::4, :], (img_width, img_height)), 
                ref_img, 
                depth_map
            )
            ssim_val = ssim(
                cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), 
                cv2.cvtColor(restored_img, cv2.COLOR_BGR2GRAY), 
                data_range=256
            )
            print("[Metric %d] PSNR: %f, %f, %f %f, SSIM: %f, fill pct: %f" % (
                i, exp_psnr_val, act_psnr_val, resize_x2_psnr_val, resize_x4_psnr_val, ssim_val, fill_pct
            ))
            total_exp_psnr.append(exp_psnr_val)
            total_act_psnr.append(act_psnr_val)
            total_resize_x2_psnr.append(resize_x2_psnr_val)
            total_resize_x4_psnr.append(resize_x4_psnr_val)
            total_ssim.append(ssim_val)
            total_fill_pct.append(fill_pct)

        prev_pcd = copy.deepcopy(pcd)

    print("[Final] PSNR: %f, %f, %f, %f, SSIM: %f, fill pct: %f" % (
        np.mean(total_exp_psnr), 
        np.mean(total_act_psnr), 
        np.mean(total_resize_x2_psnr), 
        np.mean(total_resize_x4_psnr), 
        np.mean(total_ssim), 
        np.mean(total_fill_pct)
    ))


if __name__ == '__main__':
    main()