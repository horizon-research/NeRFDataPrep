_base_ = './default.py'

expname = 'dvgo_bonsai_0_2'
basedir = './logs/bonsai_0_2'

data = dict(
    datadir='../../../bonsai/',
    dataset_type='metashape',
    white_bkgd=True,
)

