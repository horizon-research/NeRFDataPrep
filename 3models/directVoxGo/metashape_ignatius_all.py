_base_ = './default.py'

expname = 'dvgo_ignatius_0_2_all'
basedir = './logs/ignatius_0_2_all'

data = dict(
    datadir='../../../ignatius/',
    dataset_type='metashape',
    white_bkgd=True,
)

