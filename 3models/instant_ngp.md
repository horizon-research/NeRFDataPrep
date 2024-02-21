
# Instant NGP
## 1. Clone and set up their code. (see their repo)
```bash
git clone https://github.com/NVlabs/instant-ngp
cd instant-ngp/
git submodule update --init --recursive
# build gui application
xhost +
cmake ./ -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j 32
```

## 2. Changes to make:
- aabb_scale : it is decided in the blender_format data generation. I used 1 since the data is normalized.
- need to change line 284 in ```scripts/run.py``` to fix filename problem:
```python
# cam_matrix = f.get("transform_matrix", f["transform_matrix_start"])
cam_matrix = f["transform_matrix"]
```
## 3. Below is the training / Evaluation code we use


Note: We can increase network capacity to improve PSNR  but in order to stay consistent with our hardware evaluation for sythetic NeRF, we use default setting.

```bash
# training,
cd scripts/
python3 run.py ../configs/nerf/base.json --scene "$workspace"/transforms_train.json --save_snapshot "$workspace"/ingp_256_35000_base.ingp --n_steps 35000 --marching_cubes_res 256
# python3 run.py ../configs/nerf/base.json --scene ../../../../garden/transforms_train.json --save_snapshot ../../../../garden/ingp_256_35000_base_all.ingp --n_steps 35000 --marching_cubes_res 256

# testing and save snapshots of val set
python3 run.py --scene "$workspace"/transforms_train.json --load_snapshot  "$workspace"/ingp_256_35000_base.ingp --test_transforms "$workspace"/transforms_val.json --screenshot_transforms "$workspace"/transforms_val.json --screenshot_dir "$workspace"/ingp_256_35000_base_snapshots --marching_cubes_res 256

#python3 run.py --scene ../../../../garden/transforms_train.json --load_snapshot  ../../../../garden/ingp_256_35000_base_all.ingp --test_transforms ../../../../garden/transforms_val.json --screenshot_transforms ../../../../garden/transforms_val.json --screenshot_dir ../../../../garden/ingp_256_35000_base_all_snapshots --marching_cubes_res 256
```

