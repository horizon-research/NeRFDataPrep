# NeRF data preparation from real-world images 

This repo is used to transform real-world images to the form that needed by our paper [Cicero: Real-Time Neural Rendering by Radiance Warping and Memory Optimizations](), it includes five steps:
- Use [Metashape](https://www.agisoft.com/) to reconstruct mesh and camera poses from real world data
- Prepare the images dataset
- Post-processing the mesh by cropping out the background
- Use the cropped mesh to generate foreground mask and depth maps of images
- Transform the metashape data to blender dataformat that is compatible with three methods used in our paper
- Tune the parameters for real-world dataset in three methods and get the final result.

## 1. Use metashape to generate mesh and camera poses from real world data.

### (1.1) Download the software and set it up. There is a one month free trial.
### (1.2) Reconstruct the camera poses mesh following their [Manual](https://www.agisoft.com/pdf/metashape_2_1_en.pdf)
- This step mainly include two steps: (1) align photos (2) Create model (mesh)
    - align photos: we use default setting
    - Create model (mesh): we change quality to high 
- after this step, export the model (mesh in .obj format) and cameras parameters (include extrinsicts and intrinsicts) to a folder. The folder should looks like below. mesh.obj and mesh.mtl are from mesh, and meta.xml describe the camera extrinsicts and intrinsicts.
```
.
├── mesh.mtl
├── mesh.obj
└── meta.xml
```
- Create a workspace for the dataset, and in the following step we will store and load things from it. For example, create a folder called garden/ as workspace. Then put above three files inside it.

```bash
mkdir <path_to_workspace>
export workspace=<abs_path_to_workspace>
```

## 2. Prepare the images dataset

In this step we need to prepare the images for later use. The image should have extension name ".JPG". This is fine for 360 dataset, but we need to modify Tank&Temple dataset for this rule.

run:
```bash
python3 transfom_tt_images.py --in_folder trunk/images/ --out_folder trunk/images_2 --downsample_factor 2
```

In our experience, we use downsampled 4x images in 360 dataset and we downsample 2x the Tank&Temple dataset. Just put these images inside the workspace.

## 3. Post-processing the mesh by cropping out the background
 Due to sparse sampling, metashape can't reconstruct background mesh well, it well cause holes or inaccuracy in depth maps, so here we delete the background mesh that we don't care. During the experiments, we only computing sparsity in the foreground. 
 This step has two stages, first you need to decide the foreground bounding box, then you need to process the whole mesh to filter out faces outside of the bounding box.

## (3.1) Parse the metashape data
run:
```bash
export workspace=<abs_path_to_workspace>
cd crop_foreground
python3 parse_cameras_meta.py --meta_file "$workspace"/meta.xml --output_path "$workspace"/parsed_meta.pkl
# eg. python3 parse_cameras_meta.py --meta_file ../garden/meta.xml --output_path ../garden/parsed_meta.pkl

```

 ### (2.1) Decide the foreground bounding box

 use following script to visualize bounding box and mesh, adjust the bounding box to 
 make it contain only the forground, use the coordinate drawn in the viewer to help you adjust it.

 A good bounding box should: 
 - contain everything inside the camera array (prevent inconsistency between views)
 - contain the foreground using size as small as possible.
 ```bash
 cd crop_foreground
 python3 bounding_box_drawer.py --input_mesh "$workspace"/mesh.obj --bbox <path_to_bbox.txt> --parsed_meta "$workspace"/parsed_meta.pkl
 # eg. python3 bounding_box_drawer.py --input_mesh ../garden/mesh.obj --bbox ./garden_bbox.txt --parsed_meta ../garden/parsed_meta.pkl
 # see crop_foreground/garden_bbox.txt to know how to write bbox.txt
 # cx, cy, cz are centers
 # rx, ry, rz are rotation in degrees
 # lx, ly, lz are lengths of the bbox
 ```
 here is an example of adjusted bbox in garden scene:
<p float="left">
  <img src="imgs/bbox.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>

 ### (2.2) Filter out the back ground mesh outside the bounding box
 After setting the foreground region, we need to filter out the background meshes, and during our evaluation, those pixels correspond to no mesh (background pixels) won't be counted.
 run below code to filter out the background mesh:
```bash
 cd crop_foreground
python3 background_mesh_filter.py --input_mesh "$workspace"/mesh.obj --output_path "$workspace"/mesh_cut.obj --bbox <path_to_bbox> --num_workers 8
# eg. python3 background_mesh_filter.py --input_mesh ../garden/mesh.obj --output_path ../garden/mesh_cut.obj --bbox garden_bbox.txt --num_workers 8
```
After the filtering, pyrender viewer will show the bounding box and cropped result like below:
<p float="left">
  <img src="imgs/cut_mesh.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>


# 3. Normalize Camera poses and Mesh and Fix the Camera poses
Since some methods may expect foreground to be at origin and have small size, here we need to normalize the camera poses and mesh using the bounding box information in case some of them have no auto-detection and normalizion.

## (3.1) Normalize Mesh and Camera Poses using BBox information

In this stage we normalize the foreground to 1x1x1 bounding box around origin using the foreground bounding box information.
run:
```bash
python3 norm_poses_mesh.py --parsed_meta "$workspace"/parsed_meta.pkl --input_mesh "$workspace"/mesh_cut.obj --output_mesh_path "$workspace"/norm_mesh.obj --output_meta_path "$workspace"/norm_meta.pkl --bbox <path_to_bbox.txt>
#eg. python3 norm_poses_mesh.py --parsed_meta "$workspace"/parsed_meta.pkl --input_mesh "$workspace"/mesh_cut.obj --output_mesh_path "$workspace"/norm_mesh.obj --output_meta_path "$workspace"/norm_meta.pkl --bbox ../crop_foreground/trunk_bbox.txt
```
Then you will see a visualization windows shows the normalized results like below, make sure postive z-axis (blue) is pointed to the target and mesh is alighed with the axis in the same way as it aligh with the foreground bounding box. 
<p float="left">
  <img src="imgs/norm.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>

## (3.2)  Fix the camera pose by rotate it
Since camera in pyrender and blender format data all target the object using negative z-axis which is different from metashape, we need to rotate it here.
run:
```bash
python3 fix_poses.py --in_meta "$workspace"/norm_meta.pkl  --output_path "$workspace"/fix_norm_meta.pkl
```




# 4. Use the cropped mesh to generate foreground mask and depth maps of images from mesh

## (4.1) Get depth and foreground mask from mesh 
Run below code to extract depth and mask from mesh.
- set 4x downsampled for 360
- set 2x for Tank&Temple

The depth map is fp32 and will be named according to the corresponding image, and the mask is computed using depth>0, saved in np.uint8 format, also named according to the corresponding image.
```bash 
export downsample_factor=2
cd generate_depths_and_mask
python3 get_depth_and_mesh.py --norm_mesh "$workspace"/norm_mesh.obj --parsed_meta "$workspace"/fix_norm_meta.pkl --downsampled_factor "$downsample_factor" --output_folder "$workspace"/depths_masks_"$downsample_factor"
# eg. python3 get_depth_and_mesh.py --norm_mesh ../garden/norm_mesh.obj --parsed_meta ../garden/fix_norm_meta.pkl --downsampled_factor 4 --output_folder ../garden/depths_masks_4
```

## (4.2) Validate the gernerated depth and mask
To validate the depth and mask, we can overlap them with RGB image.
run:
```bash
cd generate_depths_and_mask
python3 validate.py --depth_masks_folder "$workspace"/depths_masks_"$downsample_factor" --rgb_folder "$workspace"/images_"$downsample_factor" --output_folder "$workspace"/depths_masks_"$downsample_factor"_validation
# eg. python3 validate.py --depth_masks_folder ../garden/depths_masks_4/ --rgb_folder ../garden/images_4/ --output_folder ../garden/depth_mask_4_validation
```
output will look like: (left is depth validation image, right is mask validation image.)
<p float="left">
  <img src="imgs/depth_val.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
  <img src="imgs/mask_val.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>




# 5. transform the metashape data to blender dataformat that is compatible with three methods used in our paper

## (5.1) Generate RGBA format masked image 
We use A=0 to tell the background pixels, same as blender dataset
run:
```bash
cd generate_blender_format
python3 generate_mask_image_set.py --depth_masks_folder "$workspace"/depths_masks_"$downsample_factor"/ --rgb_folder "$workspace"/images_"$downsample_factor"/ --output_folder "$workspace"/images_"$downsample_factor"_mask
# python3 generate_mask_image_set.py --depth_masks_folder ../garden/depths_masks_4/ --rgb_folder ../garden/images_4/ --output_folder ../garden/images_4_mask
```


## (5.2) Generate blender format .json meta file.
- Run below code to generate train, val and test splits.Since I have normalize the data, aabb_scale=1 works fine in my case. And I use downscale_factor=4 which will be applied to camera intrinsicts.
```bash
cd gnerate_blender_format
export aabb_scale=1
bash ./gnerate_blender_format_trainval.sh $aabb_scale "$workspace"/fix_norm_meta.pkl "$workspace" "$workspace"/images_"$downsample_factor"_mask/ "$downsample_factor"
# modified from colmap2nerf in https://github.com/NVlabs/instant-ngp
# eg. bash ./gnerate_blender_format_trainval.sh 1 ../garden/fix_norm_meta.pkl ../garden/ ../garden/images_4_mask/ 4.0
```
You shoud see "transforms_xxx.json" under the output_folder now.

- Also, we provide another split way, that is using all data for training and evaluation, it is more sensible for the comparison between our experiments and baseline, see explaination in step 6. 

Run below code to do such splitting:
```bash
cd gnerate_blender_format
bash ./gnerate_blender_format_all.sh $aabb_scale "$workspace"/fix_norm_meta.pkl "$workspace" "$workspace"/images_"$downsample_factor"_mask/ "$downsample_factor"
# modified from colmap2nerf in https://github.com/NVlabs/instant-ngp
# eg. bash ./gnerate_blender_format_all.sh 1 ../garden/fix_norm_meta.pkl ../garden/ ../garden/images_4_mask/ 4.0
```

# 5. tune the parameters for real-world dataset in three methods and get the final result.

The three methods we use in the paper include:
- [Instant NGP](https://github.com/NVlabs/instant-ngp)
- [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO)
- [TensoRF](https://github.com/apchenstu/TensoRF)

Since we are using our own dataset constructed by metashape, we need to do two things:
- Integrate our blender format data into three methods.
- tune parameters by ourselve, mainly the bounding box of nerf algorithm.

Here I will only show the results.
For details about how to integrate and tune the parameters, see Readmes in [3models](./3models) introduce the integration method and tuned parameters.
## Results 
### Split 1: Training set for training, validation set for evaluation.
- PSNR 

    | method \ dataset | 360-Garden | 360-bonsai | Tanks&Temple-Ignatius | Tanks&Temple-Ignatius-long |
    |----------|----------|----------|----------|
    | Instant NGP | **32.54** | **32.05** | 27.83 | 27.82
    | DirectVoxGo   | 30.20 | 27.56  | 27.82 | 28.944
    | Tensor RF   | 31.82 |  30.48 | **28.44** | 29.53


### Split 2: Use all train+val set for training and evaluation.
- PSNR 

    | method \ dataset | 360-Garden | 360-bonsai | Tanks&Temple-Ignatius |
    |----------|----------|----------|----------|
    | Instant NGP | **33.52** | **32.46**  | 29.39 |
    | DirectVoxGo   | 31.69 | 28.87  | 30.53 | 
    | Tensor RF   | 32.82 | 31.99 |  **31.07** |


# 6 Warping 

## PSNR: foreground only, skip=7
ignatius:
```
instant ngp: 
python3 warping_evaluation.py --nerf_results_folder ignatius/ingp_256_35000_base_snapshots_all --gt_folder ignatius/images_2 --depth_and_mask_folder ignatius/depths_masks_2 --result_path ignatius/ --item_name ignatius --meta_data_path ignatius/fix_norm_meta.pkl --downscale_factor 2 

[Final] PSNR: 29.847098, 30.205088, 37.422740, 36.840367, fill pct: 0.265401
```

garden:
```
instant ngp: 
python3 warping_evaluation.py --nerf_results_folder garden/ingp_256_35000_base_all_snapshots/ --gt_folder garden/images_4 --depth_and_mask_folder garden/depths_masks_4 --result_path garden/ --item_name garden --meta_data_path garden/fix_norm_meta.pkl --downscale_factor 4
[Final] PSNR: 31.697880, 33.262846, 36.563756, 35.389912, fill pct: 0.339527

```

bonsai:
```
instant ngp: 
python3 warping_evaluation.py --nerf_results_folder bonsai/ingp_256_35000_base_snapshots_all/ --gt_folder bonsai/images_4 --depth_and_mask_folder bonsai/depths_masks_4 --result_path garden/ --item_name garden --meta_data_path bonsai/fix_norm_meta.pkl --downscale_factor 4 
```


