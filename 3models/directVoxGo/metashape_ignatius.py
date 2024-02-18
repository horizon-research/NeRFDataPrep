_base_ = './default.py'

expname = 'dvgo_ignatius_0_2'
basedir = './logs/ignatius_0_2'

data = dict(
    datadir='../../../ignatius/',
    dataset_type='metashape',
    white_bkgd=True,
)

