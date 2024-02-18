_base_ = './default.py'

expname = 'dvgo_bonsai_0_2_all'
basedir = './logs/bonsai_0_2_all'

data = dict(
    datadir='../../../bonsai/',
    dataset_type='metashape',
    white_bkgd=True,
)

