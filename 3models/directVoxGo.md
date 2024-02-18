

# DirectVoxGO

## 1. Clone and set up their code. (see their repo)
```bash
git clone https://github.com/sunset1995/DirectVoxGO
```
## 2. Changes to make:

### (2.1) Add below codes to  ```DirectVoxGO/lib/load_data.py``` line 134
- Add metashape dataset type
It mainly add a new dataset type called metashape based on blender dataset, difference is we set ```near, far = 0., 2.```
```python
    elif args.dataset_type == 'metashape':
        images, poses, render_poses, hwf, i_split = load_metashape_data(args.datadir, args.half_res, args.testskip)
        print('Loaded Metashape', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 0., 2.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]  
```
- load metashape dataset type

add ```from .load_metashape import load_metashape_data``` at the head.


### (2.2)  Copy our [load_meatashape.py](./directVoxGo/load_meatashape.py) in [3models/directVoxGo/](./directVoxGo/) to their lib/ folder
It is copied from load _blender.py but change line 57 to match our filename. Also the function name is changed.
```python
            # fname = os.path.join(basedir, frame['file_path'] + '.png')
            fname = os.path.join(basedir, frame['file_path'])
```

### (2.2)   Put our config file [metashape.py](./directVoxGo/metashape.py) in [3models/directVoxGo/](./directVoxGo/) to their configs folder, then you can run the training code:

The config is same as blender but change the name of exp, output and dataset.


## 3. Below is the training / Evaluation code we use

```bash
python3 run.py --config configs/metashape.py --render_test
```
