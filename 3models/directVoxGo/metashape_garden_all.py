_base_ = './default.py'

expname = 'dvgo_garden_0_2_all'
basedir = './logs/garden_0_2_all'

data = dict(
    datadir='../../../garden/',
    dataset_type='metashape',
    white_bkgd=True,
)

