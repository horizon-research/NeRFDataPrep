import numpy as np 
import cv2
import math
import open3d as o3d
import copy
import json
from skimage.metrics import structural_similarity as ssim
from skvideo.io import FFmpegWriter
import argparse
from glob import glob
import pickle
from tqdm import tqdm
# FocalLenthDict = {
#     "lego" : 1111,
#     "mic" : 1111,
#     "drums": 1250,
#     "hotdog": 1111,
#     "chess" : 1250,
#     "chair" : 1111,
#     "kitchen" : 1111,
#     "mic" : 1111,
#     "room" : 1111,
#     "ship" : 1111,
# }

# def PSNR(original, compressed, depth_map):
#     mse = np.mean((original[depth_map>0] - compressed[depth_map>0]) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
#     return psnr


def PSNR(original, compressed, mask = None):
    if mask is not None:
        mse = np.mean((original[mask] - compressed[mask]) ** 2)
    else:
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

def pc_to_rgb(pcd, fl_x, fl_y, cx, cy, img_width, img_height, depth_range):
    restored_img = np.zeros((img_height, img_width, 3))
    z_buffer = np.zeros((img_height, img_width))
    z_buffer[:, :] = depth_range
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # np.save("pcd.npy", points)
    # np.save("rgb.npy", colors)

    for i, pt in enumerate(points):
        rgb = colors[i]

        if np.sum(rgb) == 0:
            continue

        x, y, z = pt
        z = np.abs(z) # needs to negate z value
        x_coord = int(cx + x / z * fl_x - 0.5)
        y_coord = int(cy - y / z * fl_y - 0.5)
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
            if ref_depth[r, c] > 0 and ref_depth[r, c] < z_buffer[r, c] * 0.995:             
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

def find_reference_frame(index, skip_count, total_cnt):
    batch = index // skip_count
    return min(batch * skip_count + skip_count//2, total_cnt-1)

def option():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument(
        "--nerf_results_folder",
        type=str,
        default="synthetic_dataset",
        help="path to nerf_results_folder",
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        default="synthetic_dataset",
        help="path to gt_folder",
    )
    parser.add_argument(
        "--depth_and_mask_folder",
        type=str,
        default="synthetic_dataset",
        help="path to depth_and_mask_folder",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="path to result, default: None",
    )
    parser.add_argument(
        "--item_name",
        type=str,
        default=None,
        help="evaluated item name, default: None",
    )
    parser.add_argument(
        "--skip_count", type=int, help="the number of frames skipped inference", default=7
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="preview the result, default: False",
    )
    parser.add_argument(
        "--meta_data_path",
        type=str,
        default="meta_data_path",
        help="path to meta_data_path contains camera params",
    )
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default="downscale_factor",
        help="downscale_factor",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="method_name",
        choices=["temp_instant_ngp", "temp_dgo", "temp_tensorrf"],
        help="eg. Cicero + instant_igp, tensorrf...",
    )

    # parse
    args = parser.parse_args()

    return args

def main():

    args = option()
    # read meta dat 
    with open(args.meta_data_path, "rb") as f:
        meta_data = pickle.load(f)

    fl_x = meta_data["intrinsics"]["f"] / args.downscale_factor
    fl_y = meta_data["intrinsics"]["f"] / args.downscale_factor
    cx = meta_data["intrinsics"]["cx"] / args.downscale_factor
    cy = meta_data["intrinsics"]["cy"] / args.downscale_factor
    img_width = round(meta_data["intrinsics"]["width"] / args.downscale_factor)
    img_height = round(meta_data["intrinsics"]["height"] / args.downscale_factor)

    # dataset config
    item_name = args.item_name + "_" + args.method_name

    # count number of frames
    skip_count = args.skip_count
    total_cnt = len(meta_data["name_poses"])      # total frame count


    # init and stats.
    prev_pcd = None
    total_exp_psnr_all = []
    total_act_psnr_all = []
    total_resize_x2_psnr_all = []
    total_resize_x4_psnr_all = []
    total_resize_2x4_psnr_all = []
    total_exp_psnr_fg = []
    total_act_psnr_fg = []
    total_resize_x2_psnr_fg = []
    total_resize_x4_psnr_fg = []
    total_resize_2x4_psnr_fg = []
    
    # total_ssim = []
    total_fill_pct = []
    not_ref_fill_pct = []

    # output log file and write a demo video
    log_file = open("%s/%s_s%02d_log.txt" % (args.result_path, item_name, args.skip_count), "w")
    writer = FFmpegWriter(
        "%s/%s_demo_s%02d.mp4" % (args.result_path, item_name, args.skip_count),
        inputdict={
            '-r': str(3),
        },
        outputdict={
            '-r': str(3),
        }
    )
    prev_img = None
    for i, name_pose in tqdm(enumerate(meta_data["name_poses"]), total=total_cnt):
        if i % skip_count == 0:
            ref_num = i
        else:
            ref_num = i-1

        name = name_pose["name"]
        pose = name_pose["transform"]
        curr_depth_fn = args.depth_and_mask_folder + "/" + name + "_depth_fp32.npy"
        curr_mask_fn = args.depth_and_mask_folder + "/" + name + "_mask_uint8.npy"
        # import ipdb; ipdb.set_trace()
        if args.method_name == "temp_instant_ngp":
            curr_rgb_fn = args.nerf_results_folder + "/" + name + ".png"
        else:
            formatted_string = "{:03d}".format(i)
            curr_rgb_fn = args.nerf_results_folder + "/" + formatted_string + ".png"



        ref_depth_fn = args.depth_and_mask_folder + "/" + meta_data["name_poses"][ref_num]["name"] + "_depth_fp32.npy"
        ref_mask_fn = args.depth_and_mask_folder + "/" + meta_data["name_poses"][ref_num]["name"] + "_mask_uint8.npy"
        if args.method_name == "temp_instant_ngp":
            ref_rgb_fn = args.nerf_results_folder + "/" + meta_data["name_poses"][ref_num]["name"] + ".png"
        else:
            formatted_string = "{:03d}".format(ref_num)
            ref_rgb_fn = args.nerf_results_folder + "/" + formatted_string + ".png"

        gt_rgb_fn = args.gt_folder + "/" + name + ".JPG"
        


        print(curr_rgb_fn, ref_rgb_fn, gt_rgb_fn)


        # load current depth and image data
        curr_trans_mat = pose
        inverse_trans_mat = inverse_transform_mat(curr_trans_mat)
        curr_depth_map = np.load(curr_depth_fn)
        # add a channel
        curr_depth_map = curr_depth_map[:, :, np.newaxis]
        curr_mask = np.load(curr_mask_fn)

        curr_img = cv2.imread(curr_rgb_fn)

        # load reference depth and image data
        ref_trans_mat = meta_data["name_poses"][ref_num]["transform"]
        ref_depth_map = np.load(ref_depth_fn)
        ref_mask = np.load(ref_mask_fn)
        # add a channel
        ref_depth_map = ref_depth_map[:, :, np.newaxis]

        if i % skip_count == 0:
            ref_img = cv2.imread(ref_rgb_fn)
        else:
            ref_img = prev_img

        # load ground truth image
        gt_img = cv2.imread(gt_rgb_fn)

        # mask, act, ref gt
        curr_mask = curr_mask[:, :, np.newaxis] == 1
        # make it 3 channel
        curr_mask = np.repeat(curr_mask, 3, axis=2)
        ref_mask = ref_mask[:, :, np.newaxis] == 1
        # make it 3 channel
        ref_mask = np.repeat(ref_mask, 3, axis=2)
        curr_img = curr_img * curr_mask
        ref_img = ref_img * ref_mask
        gt_img = gt_img * curr_mask



        if ref_num != i:
            # conpose point cloud
            curr_pcd = compose_pcd(curr_depth_map, np.array(curr_img), fl_x, fl_y, cx, cy)
            # transform point cloud from camera coordinate to world coordinate
            # so that all point clouds will be at the same coordinate system
            curr_pcd = curr_pcd.transform(curr_trans_mat)

            # conpose point cloud
            ref_pcd = compose_pcd(ref_depth_map, np.array(ref_img), fl_x, fl_y, cx, cy)
            # transform point cloud from camera coordinate to world coordinate
            # so that all point clouds will be at the same coordinate system
            ref_pcd = ref_pcd.transform(ref_trans_mat)

            # visualize 3D point cloud
            # visualize_pcd(
            #     [
            #         copy.deepcopy(curr_pcd),    # previous transformed pcd
            #         copy.deepcopy(ref_pcd)      # current pcd
            #     ],
            #     img_width,
            #     img_height
            # )

            restored_img, z_buffer = pc_to_rgb(
                copy.deepcopy(ref_pcd).transform(inverse_trans_mat), 
                fl_x, fl_y, cx, cy, 
                img_width, img_height, depth_range=1000
            )

            fill_pct = 0
            restored_img, fill_pct = fill_disocclusion(
                restored_img, np.array(curr_img), curr_depth_map[:, :, 0], z_buffer
            )
        else:
            # assign restored image to be the refernce image
            fill_pct = 1    # render full-size image
            restored_img = np.array(curr_img)

        # compute the diff between restored image and ground truth image
        diff = np.abs(restored_img.astype(np.float32) - gt_img.astype(np.float32)).astype(np.uint8)
        comb = np.hstack((restored_img, gt_img, diff))

        prev_img = copy.deepcopy(restored_img)

        # visualize result
        if args.visualize:
            cv2.imshow("restored", comb)
            cv2.waitKey(10)

        # write result 
        writer.writeFrame(
            cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
        )
        
        # evaluate accuracy
        # import ipdb; ipdb.set_trace()

        # compute all PSNR
        exp_psnr_val_all = PSNR(restored_img, gt_img)
        act_psnr_val_all = PSNR(curr_img, gt_img)
        resize_x2_psnr_val_all = PSNR(
            cv2.resize(curr_img[::2, ::2, :], (img_width, img_height)), 
            gt_img
        )
        resize_x4_psnr_val_all = PSNR(
            cv2.resize(curr_img[::4, ::4, :], (img_width, img_height)), 
            gt_img
        )
        resize_2x4_psnr_val_all = PSNR(
            cv2.resize(curr_img[::2, ::4, :], (img_width, img_height)), 
            gt_img
        )
        print(
            "[Metric %d] ALL PSNR: %f, %f, %f, %f, %f, fill pct: %f" % (
                i, exp_psnr_val_all, act_psnr_val_all, resize_x2_psnr_val_all, 
                resize_x4_psnr_val_all, resize_2x4_psnr_val_all, fill_pct
            )
        )
        log_file.write(
            "[Metric %d] ALL PSNR: %f, %f, %f, %f, %f, fill pct: %f\n" % (
                i, exp_psnr_val_all, act_psnr_val_all, resize_x2_psnr_val_all, 
                resize_x4_psnr_val_all, resize_2x4_psnr_val_all, fill_pct
            )
        )

        total_exp_psnr_all.append(exp_psnr_val_all)
        total_act_psnr_all.append(act_psnr_val_all)
        total_resize_x2_psnr_all.append(resize_x2_psnr_val_all)
        total_resize_x4_psnr_all.append(resize_x4_psnr_val_all)

        # compute fg PSNR
        exp_psnr_val_fg = PSNR(restored_img, gt_img, curr_mask)
        act_psnr_val_fg = PSNR(curr_img, gt_img, curr_mask)
        resize_x2_psnr_val_fg = PSNR(
            cv2.resize(curr_img[::2, ::2, :], (img_width, img_height)), 
            gt_img, 
            curr_mask
        )
        resize_x4_psnr_val_fg = PSNR(
            cv2.resize(curr_img[::4, ::4, :], (img_width, img_height)), 
            gt_img, 
            curr_mask
        )
        resize_2x4_psnr_val_fg = PSNR(
            cv2.resize(curr_img[::2, ::4, :], (img_width, img_height)), 
            gt_img, 
            curr_mask
        )
        print(
            "[Metric %d] FG PSNR: %f, %f, %f, %f, %f, fill pct: %f" % (
                i, exp_psnr_val_fg, act_psnr_val_fg, resize_x2_psnr_val_fg, 
                resize_x4_psnr_val_fg, resize_2x4_psnr_val_fg, fill_pct
            )
        )
        log_file.write(
            "[Metric %d] FG PSNR: %f, %f, %f, %f, %f, fill pct: %f\n" % (
                i, exp_psnr_val_fg, act_psnr_val_fg, resize_x2_psnr_val_fg, 
                resize_x4_psnr_val_fg, resize_2x4_psnr_val_fg, fill_pct
            )
        )

        total_exp_psnr_fg.append(exp_psnr_val_fg)
        total_act_psnr_fg.append(act_psnr_val_fg)
        total_resize_x2_psnr_fg.append(resize_x2_psnr_val_fg)
        total_resize_x4_psnr_fg.append(resize_x4_psnr_val_fg)
        


        total_fill_pct.append(fill_pct)
        if ref_num != i:
            not_ref_fill_pct.append(fill_pct)


    print(
        "[Final] ALL PSNR: %f, %f, %f, %f, %f all fill pct: %f, not_ref_fill_pc: %f" % (
            np.mean(total_exp_psnr_all), 
            np.mean(total_act_psnr_all), 
            np.mean(total_resize_x2_psnr_all), 
            np.mean(total_resize_x4_psnr_all),
            np.mean(total_resize_2x4_psnr_all),
            np.mean(total_fill_pct),
            np.mean(not_ref_fill_pct)
        )
    )

    print(
        "[Final] FG PSNR: %f, %f, %f, %f, %f fill pct: %f, not_ref_fill_pc: %f" % (
            np.mean(total_exp_psnr_fg), 
            np.mean(total_act_psnr_fg), 
            np.mean(total_resize_x2_psnr_fg), 
            np.mean(total_resize_x4_psnr_fg),
            np.mean(total_resize_2x4_psnr_all),
            np.mean(total_fill_pct),
            np.mean(not_ref_fill_pct)
        )
    )

    log_file.write(
        "[Final] ALL PSNR: %f, %f, %f, %f, %f fill pct: %f, not_ref_fill_pc: %f\n" % (
            np.mean(total_exp_psnr_all), 
            np.mean(total_act_psnr_all), 
            np.mean(total_resize_x2_psnr_all), 
            np.mean(total_resize_x4_psnr_all),
            np.mean(total_resize_2x4_psnr_all),
            np.mean(total_fill_pct),
            np.mean(not_ref_fill_pct)
        )   
    )

    log_file.write(
        "[Final] FG PSNR: %f, %f, %f, %f,%f fill pct: %f, not_ref_fill_pc: %f\n" % (
            np.mean(total_exp_psnr_fg), 
            np.mean(total_act_psnr_fg), 
            np.mean(total_resize_x2_psnr_fg), 
            np.mean(total_resize_x4_psnr_fg),
            np.mean(total_resize_2x4_psnr_all),
            np.mean(total_fill_pct),
            np.mean(not_ref_fill_pct)
        )
    )
    writer.close()
    log_file.close()


if __name__ == '__main__':
    main()