from glob import glob

index = '0614'
rotate = '_rotate'
crop = ''

raw_root = '/home1/hanxy/hanxy_data/raw_data_{}{}'.format(index, rotate)
rotate_root = '/home1/hanxy/hanxy_data/raw_data_{}_rotate'.format(index, rotate)
lung_root = './lung_{}{}{}'.format(index, rotate, crop)
lesion_root = './lesion_{}{}{}'.format(index, rotate, crop)
meta_root = '/home1/hanxy/hanxy_data/meta_data_{}'.format(index)

csv_name = 'total_data_{}{}{}.csv'.format(index, rotate, crop)

npy_path = glob(r'/home1/hanxy/hanxy_data/raw_data_{}{}/*.npy'.format(index, rotate))

sample_times = 10
uncerten_root = './uncerten_{}{}{}'.format(index, rotate, crop)
