_base_ = './default.py'

expname = 'dvgo_ignatius_0_2_long'
basedir = './logs/ignatius_0_2_long'

data = dict(
    datadir='../../../ignatius_900_10_seq/',
    dataset_type='metashape',
    white_bkgd=True,
)

