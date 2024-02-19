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

FocalLenthDict = {
    "lego" : 1111,
    "mic" : 1111,
    "drums": 1250,
    "hotdog": 1111,
    "chess" : 1250,
    "chair" : 1111,
    "kitchen" : 1111,
    "mic" : 1111,
    "room" : 1111,
    "ship" : 1111,
}

# def PSNR(original, compressed, depth_map):
#     mse = np.mean((original[depth_map>0] - compressed[depth_map>0]) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
#     return psnr


def PSNR(original, compressed):
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
    np.save("pcd.npy", points)
    np.save("rgb.npy", colors)

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
        "--dataset_path",
        type=str,
        default="synthetic_dataset",
        help="path to synthetic dataset, default: 'synthetic_dataset'",
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
        "--split",
        type=str,
        default="test",
        help="evaluated train/test set, default: test",
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
        "--image_shape",
        nargs=2,
        type=int,
        default=[800, 800],
        help="image shape, [HEIGHT, WIDTH] default: [800, 800]",
    )
    parser.add_argument(
        "--focal_length",
        nargs=2,
        type=int,
        default=[1250, 1250],
        help="camera focal length in x- and y-axis, [HEIGHT, WIDTH] default: [1250, 1250]",
    )
    parser.add_argument(
        "--depth_scale", 
        type=int, 
        default=0.1, 
        help="the depth map scale, the depth values are scaled by this value, e.g. `0.1`",
    )

    # parse
    args = parser.parse_args()

    return args

def main():

    args = option()

    # camera configs
    if args.item_name not in FocalLenthDict:
        fl_x = args.focal_length[0]         # x-axis focal length
        fl_y = args.focal_length[1]         # y-axis focal length
    else:
        fl_x = FocalLenthDict[args.item_name]   # x-axis focal length
        fl_y = FocalLenthDict[args.item_name]   # y-axis focal length

    img_width = args.image_shape[0]     # image width
    img_height = args.image_shape[1]    # image height
    cx = img_width/2                    # camera center in x-axis
    cy = img_height/2                   # camera center in y-axis
    depth_scale = args.depth_scale
    depth_range = 1/depth_scale

    # dataset config
    item_name = args.item_name
    split = args.split
    depth_dir = "%s/%s/depth" % (args.dataset_path, item_name)
    img_dir = "%s/%s_result" % (args.result_path, item_name)
    gt_dir = "%s/%s/%s" % (args.dataset_path, item_name, split)
    transform_fn = "%s/%s/transforms_%s.json" % (args.dataset_path, item_name, split)

    # count number of frames
    skip_count = args.skip_count
    total_cnt = len(glob("%s/*.exr" % depth_dir))      # total frame count

    # load trasnformation matrix files
    transform_list = load_transform_file(transform_fn)

    # init and stats.
    prev_pcd = None
    total_exp_psnr = []
    total_act_psnr = []
    total_resize_x2_psnr = []
    total_resize_x4_psnr = []
    total_ssim = []
    total_fill_pct = []

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

    for i in range(0, total_cnt, 1):
        # find current rendering file
        curr_depth_fn = "%s/r_%d.exr" % (depth_dir, i)
        # curr_fn = "%s/out_%03d.png" % (img_dir, i)              # instant-ngp
        # curr_fn = "%s/imgs_test_all/%03d.png" % (img_dir, i)    # tensort
        curr_fn = "%s/%03d.png" % (img_dir, i)                  # DirectVoxGO
        # find reference rendered file
        ref_num = find_reference_frame(i, skip_count, total_cnt)
        ref_depth_fn = "%s/r_%d.exr" % (depth_dir, ref_num)
        # ref_fn = "%s/out_%03d.png" % (img_dir, ref_num)             # instant-ngp
        # ref_fn = "%s/imgs_test_all/%03d.png" % (img_dir, ref_num)   # tensort
        ref_fn = "%s/%03d.png" % (img_dir, ref_num)                 # DirectVoxGO
        # find ground truth file
        gt_fn = "%s/r_%d.png" % (gt_dir, i)
        print(curr_fn, ref_fn, gt_fn)


        # load current depth and image data
        curr_trans_mat = transform_list[i]        
        inverse_trans_mat = inverse_transform_mat(curr_trans_mat)
        curr_depth_map = cv2.imread(curr_depth_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/depth_scale
        curr_depth_map[curr_depth_map > (depth_range-0.01)] = 0
        curr_img = cv2.imread(curr_fn)

        # load reference depth and image data
        ref_trans_mat = transform_list[ref_num]
        ref_depth_map = cv2.imread(ref_depth_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/depth_scale
        ref_depth_map[ref_depth_map > (depth_range-0.01)] = 0
        ref_img = cv2.imread(ref_fn)

        # load ground truth image
        gt_img = cv2.imread(gt_fn)

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
                img_width, img_height, depth_range
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

        # visualize result
        if args.visualize:
            cv2.imshow("restored", comb)
            cv2.waitKey(10)

        # write result 
        writer.writeFrame(
            cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
        )
        
        # evaluate accuracy
        exp_psnr_val = PSNR(restored_img, gt_img)
        act_psnr_val = PSNR(curr_img, gt_img)
        resize_x2_psnr_val = PSNR(
            cv2.resize(curr_img[::2, ::2, :], (img_width, img_height)), 
            gt_img
        )
        resize_x4_psnr_val = PSNR(
            cv2.resize(curr_img[::4, ::4, :], (img_width, img_height)), 
            gt_img
        )
        print(
            "[Metric %d] PSNR: %f, %f, %f %f, fill pct: %f" % (
                i, exp_psnr_val, act_psnr_val, resize_x2_psnr_val, 
                resize_x4_psnr_val, fill_pct
            )
        )
        log_file.write(
            "[Metric %d] PSNR: %f, %f, %f %f, fill pct: %f\n" % (
                i, exp_psnr_val, act_psnr_val, resize_x2_psnr_val, 
                resize_x4_psnr_val, fill_pct
            )
        )
        total_exp_psnr.append(exp_psnr_val)
        total_act_psnr.append(act_psnr_val)
        total_resize_x2_psnr.append(resize_x2_psnr_val)
        total_resize_x4_psnr.append(resize_x4_psnr_val)
        total_fill_pct.append(fill_pct)


    print(
        "[Final] PSNR: %f, %f, %f, %f, fill pct: %f" % (
            np.mean(total_exp_psnr), 
            np.mean(total_act_psnr), 
            np.mean(total_resize_x2_psnr), 
            np.mean(total_resize_x4_psnr),
            np.mean(total_fill_pct)
        )
    )
    log_file.write(
        "[Final] PSNR: %f, %f, %f, %f, fill pct: %f\n" % (
            np.mean(total_exp_psnr), 
            np.mean(total_act_psnr), 
            np.mean(total_resize_x2_psnr), 
            np.mean(total_resize_x4_psnr),
            np.mean(total_fill_pct)
        )   
    )
    writer.close()
    log_file.close()


if __name__ == '__main__':
    main()