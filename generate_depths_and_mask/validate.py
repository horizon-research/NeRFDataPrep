import cv2
import numpy as np
import os
from tqdm import tqdm  # Import tqdm
import re

def extract_number(filename):
    # Using regular expression to find the first sequence of digits in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0  # Return 0 if no digits are found

def overlap_depth_mask_on_rgb(depth_masks_folder, rgb_folder, output_folder, alpha=0.5):
    rgb_files = []
    depth_files = []
    mask_files = []

    for file in os.listdir(rgb_folder):
        rgb_files.append(file)
    for file in os.listdir(depth_masks_folder):
        if "depth" in file:
            depth_files.append(file)
        elif "mask" in file:
            mask_files.append(file)
        
    
    # pad the rgb name before sort
    # for i in range(len(rgb_files)):
    #     rgb_files[i] = rgb_files[i].zfill(10)

    rgb_files.sort(key=extract_number)
    depth_files.sort(key=extract_number)
    mask_files.sort(key=extract_number)

    # check number of files
    # if len(rgb_files) != len(depth_files) or len(rgb_files) != len(mask_files):
    #     print("Number of files in the folders do not match. Exiting.")
    #     return
    
    for rgb_file, depth_file, mask_file in tqdm(zip(rgb_files, depth_files, mask_files), total=len(rgb_files), desc="Processing"):
        # scale the depth image from 0-1 to 0-255
        depth_img = np.load(os.path.join(depth_masks_folder, depth_file))
        # normalize the depth image
        MAX = np.max(depth_img)
        MIN = np.min(depth_img)

        depth_img = ((depth_img - MIN) / (MAX - MIN)) * 255.0
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

        # scale the mask image from 0-1 to 0-255
        mask_img = np.load(os.path.join(depth_masks_folder, mask_file))
        mask_img = (mask_img * 255).astype(np.uint8)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        # read rgb
        rgb_img = cv2.imread(os.path.join(rgb_folder, rgb_file))
        

        # overlap the depth on the RGB image
        # import ipdb; ipdb.set_trace()
        overlap_depth = cv2.addWeighted(rgb_img, alpha, depth_img, 1 - alpha, 0)
        # overlap the mask on the depth
        overlap_mask = cv2.addWeighted(rgb_img, alpha, mask_img, 1 - alpha, 0)


        # save the overlapped image
        cv2.imwrite(os.path.join(output_folder, f"{rgb_file}_mask_overlapped.jpg"), overlap_mask)
        cv2.imwrite(os.path.join(output_folder, f"{rgb_file}_depth_overlapped.jpg"), overlap_depth)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create video from folder')
    parser.add_argument('--depth_masks_folder', type=str, help='Path to the folder containing depths and masks')
    parser.add_argument('--rgb_folder', type=str, help='Path to the folder containing RGB images')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder of overlapped results')
    parser.add_argument('--alpha', type=float, default=0.5, help='Transparency of the depth/mask overlay')
    args = parser.parse_args()
    # check if the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    overlap_depth_mask_on_rgb(args.depth_masks_folder, args.rgb_folder, args.output_folder, args.alpha)
