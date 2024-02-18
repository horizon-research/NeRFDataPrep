
# TensoRF
## 1. Clone and set up their code. (see their repo)
```bash
git clone https://github.com/apchenstu/TensoRF
```

## 2. Changes to make:

### (2.1) copy our config files to ```configs/``` folder

For example, ```metashape.txts```

### (2.1) copy our [metashape.py](./tensorRF/metashape.py) in [3models/tensorRF/](./tensorRF/) to their dataLoader/ folder

### (2.2) replace their __init__.py in dataLoader/ folder using:
```python
from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .metashape import Metashape



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'metashape':Metashape}
```

### (2.3) replace their __init__.py in dataLoader/ folder using:
add our dataset in ```opt.py``` by modifying line 29
```python
    # parser.add_argument('--dataset_name', type=str, default='blender',
    #                     choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data'])

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'metashape'])
```

## 3. Below is the training / Evaluation code we use
```bash
 python3 train.py --config configs/metashape.txt

```