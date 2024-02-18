import os
import argparse
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar

def downsample_image(input_path, output_path, downsample_factor):
    """
    Downsamples an image by the given factor and saves it to the output path.

    :param input_path: Path to the input image.
    :param output_path: Path where the downsampled image will be saved.
    :param downsample_factor: Factor by which to downsample the image.
    """
    with Image.open(input_path) as img:
        # Calculate new size
        new_size = (round( float(img.width) / float(downsample_factor)), round(float(img.height) / float(downsample_factor)))
        # Resize the image
        downsampled_img = img.resize(new_size)
        # Save the downsampled image
        downsampled_img.save(output_path)

def process_images(in_folder, out_folder, downsample_factor):
    """
    Processes all images in the input folder, downsampling them and saving the results in the output folder.

    :param in_folder: Path to the input folder containing images.
    :param out_folder: Path to the output folder where downsampled images will be saved.
    :param downsample_factor: Factor by which to downsample the images.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for i, file_name in tqdm(enumerate(os.listdir(in_folder)), total=len(os.listdir(in_folder))):
        input_path = os.path.join(in_folder, file_name)
        if os.path.isfile(input_path):
            # Change the file extension to .JPG
            base_name = os.path.splitext(file_name)[0] + '.JPG'
            output_path = os.path.join(out_folder, base_name)
            # Downsample and save the image
            downsample_image(input_path, output_path, downsample_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample images in a folder")
    parser.add_argument("--in_folder", type=str, help="Path to the input folder")
    parser.add_argument("--out_folder", type=str, help="Path to the output folder")
    parser.add_argument("--downsample_factor", type=int, help="Factor by which to downsample the images")

    args = parser.parse_args()

    process_images(args.in_folder, args.out_folder, args.downsample_factor)
