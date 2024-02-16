# NeRF data preparation from real-world images 

This repo is used to transform real-world images to the form that needed by our paper [Cicero: Real-Time Neural Rendering by Radiance Warping and Memory Optimizations](), it includes five steps:
- use [Metashape](https://www.agisoft.com/) to reconstruct mesh and camera poses from real world data
- post-processing the mesh by cropping out the region of interests
- use the cropped mesh to generate foreground mask and depth maps of images
- transform the metashape data to blender dataformat that is compatible with three methods used in our paper
- tune the parameters for real-world dataset in three methods and get the final result.

## 1. Use metashape to generate mesh and camera poses from real world data.

### Download the software and set it up. There is a one month free trial.
### Reconstruct the camera poses mesh following their [Manual](https://www.agisoft.com/pdf/metashape_2_1_en.pdf)
- This step mainly include two steps: (1) align photos (2) Create model (mesh)
    - align photos: we use default setting
    - Create model (mesh): we change quality to high 
- after this step, export the model (mesh in .obj format) and cameras parameters (include extrinsicts and intrinsicts) to a folder. The folder should looks like below. mesh.obj and mesh.mtl is from mesh, and meta.xml describe the camera extrinsicts and intrinsicts.
```
.
├── mesh.mtl
├── mesh.obj
└── meta.xml
```

## 2. Post-processing the mesh by cropping out the region of interests
 Due to sparse sampling, metashape can't reconstruct background mesh well, it well cause holes or inaccuracy in depth maps, so here we crop out the foreground mesh that we care. During the experiences of Cicero, we only render pixels, computing PSNR and sparsity in the foreground. This step has two stages, first you need to decide the foreground bounding box, then you need to process the whole mesh to filter out faces outside of the bounding box.

 ### Decide the foreground bounding box

 use following script to visualize bounding box and mesh, adjust the bounding box to 
 make it contain the forground, use the coordinate drawn in the viewer to help you adjust it.
 ```bash
 cd crop_foreground
 python3 bounding_box_drawer.py --input_mesh <path_to_mesh.obj> --bbox <path_to_bbox.txt>
 # ex: python3 bounding_box_drawer.py --input_mesh ../garden/mesh.obj --bbox ./garden_bbox.txt
 # see crop_foreground/garden_bbox.txt to know how to write bbox.txt
 # cx, cy, cz are centers
 # rx, ry, rz are rotation in degrees
 # lx, ly, lz are lengths of the bbox
 ```
 here is an example of adjusted bbox in garden scene:
<p float="left">
  <img src="bbox.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>

 ### Filter out the back ground mesh outside the bounding box
 After setting the foreground region, we need to filter out the background meshes, and during our evaluation, those pixels correspond to no mesh (background pixels) won't be counted.
 run below code to filter out the background mesh:
```bash
 cd crop_foreground
python3 background_mesh_filter.py --input_mesh <path_to_origin_mesh.obj> --output_path <path_to_save_cut_mesh.obj> --bbox <path_to_bbox.txt> --num_workers 16
# ex: python3 background_mesh_filter.py --input_mesh ../garden/mesh.obj --output_path ../garden/mesh_cut.obj --bbox garden_bbox.txt --num_workers 16
```
After the filtering, pyrender viewer will show the bounding box and cropped result like below:
<p float="left">
  <img src="cut_mesh.png" alt="Input Image" style="width: 40%; margin-right: 20px;" />
</p>


# 3. Use the cropped mesh to generate foreground mask and depth maps of images

This is for evaluation in our paper. This includes two steps: First, parse the camera data from metashape. Second, render depth and foreground mask of corresonding camera perspectives using pyrender.

## Parse the camera data from metashape
run below code, notice that we change cx, cy, and the pose to align with the blender dataset
```bash
cd generate_depths_and_mask
python3 parse_cameras_meta.py --meta_file <path_to_meta.xml> --output_path <path_to_save_parsed_meta.pkl>
# ex: python3 parse_cameras_meta.py --meta_file ../garden/meta.xml --output_path ../garden/parsed_meta.pkl
```

## Get depth and foreground mask from mesh 
run below code. Since we are testing 4x downsampled dataset, we set downsampled_factor to 4.
The depth map is fp32 and will be named according to the corresponding image, and the mask is computed using depth>0, saved in np.uint8 format, also named according to the corresponding image.
```bash 
cd generate_depths_and_mask
python3 get_depth_and_mesh.py --cut_mesh <path_to_cut_mesh.obj> --parsed_meta ../garden/<path_to_parsed_meta.pkl> --downsampled_factor 4 --output_folder <path_to_save_output.npy>
# ex: python3 get_depth_and_mesh.py --cut_mesh ../garden/mesh_cut.obj --parsed_meta ../garden/parsed_meta.pkl --downsampled_factor 4 --output_folder ../garden/depths_4
```
