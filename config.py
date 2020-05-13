from glob import glob

index = 'mild2'
rotate = '_rotate'

raw_root = '/home1/hanxy/hanxy_data/raw_data_{}{}'.format(index, rotate)
rotate_root = '/home1/hanxy/hanxy_data/raw_data_{}_rotate'.format(index, rotate)
lung_root = './lung_{}{}'.format(index, rotate)
lesion_root = './lesion_{}{}'.format(index, rotate)
meta_root = '/home1/hanxy/hanxy_data/meta_data_{}'.format(index)

csv_name = 'total_data_{}{}'.format(index, rotate)

npy_path = glob(r'/home1/hanxy/hanxy_data/raw_data_{}{}/*.npy'.format(index, rotate))
