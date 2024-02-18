_base_ = './default.py'

expname = 'dvgo_garden'
basedir = './logs/garden'

data = dict(
    datadir='../../garden/',
    dataset_type='metashape',
    white_bkgd=True,
)

