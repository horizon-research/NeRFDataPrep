import cv2
import numpy as np
import os
from tqdm import tqdm  # Import tqdm
import random
import re
def extract_number(filename):
    # Using regular expression to find the first sequence of digits in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0  # Return 0 if no digits are found

def mask_rgb_img(depth_masks_folder, rgb_folder, output_folder):
    rgb_files = []
    mask_files = []

    for file in os.listdir(rgb_folder):
        rgb_files.append(file)
    for file in os.listdir(depth_masks_folder):
        if "mask" in file:
            mask_files.append(file)

    # check number of files
    if len(rgb_files) != len(mask_files):
        print("Number of files in the folders do not match. Exiting.")
        return
    
    rgb_files.sort(key=extract_number)
    mask_files.sort(key=extract_number)
    
    for rgb_file, mask_file in tqdm(zip(rgb_files, mask_files), total=len(rgb_files), desc="Processing"):
        # scale the mask image from 0-1 to 0-255
        mask_img = np.load(os.path.join(depth_masks_folder, mask_file))
        mask_img = (mask_img).astype(np.uint8)
        mask_boolean = mask_img > 0

        # read rgb
        rgb_img = cv2.imread(os.path.join(rgb_folder, rgb_file))
        
        # Create an empty alpha channel with the same dimensions as the RGB image
        alpha_channel = np.zeros(rgb_img.shape[:2], dtype=rgb_img.dtype)

        # Set alpha to 255 (fully opaque) in regions to keep, based on the mask
        alpha_channel[mask_boolean] = 255

        # save the four channel .png image with opacity = 0 in masked region
        rgba_img = np.dstack((rgb_img, alpha_channel))

        rgba_name = rgb_file.replace(".JPG", ".png")
        # save the masked image
        cv2.imwrite(os.path.join(output_folder, rgba_name), rgba_img)



    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create video from folder')
    parser.add_argument('--depth_masks_folder', type=str, help='Path to the folder containing depths and masks')
    parser.add_argument('--rgb_folder', type=str, help='Path to the folder containing RGB images')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder of overlapped results')
    args = parser.parse_args()
    # check if the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    mask_rgb_img(args.depth_masks_folder, args.rgb_folder, args.output_folder)
