from glob import glob
from utils_mri import *
import numpy as np
# import matplotlib.pyplot as plt

from scipy.misc.pilutil import imresize  # scipy<=1.1


from utils_mri import *


def read_from_nii(nii_path=r'E:\Lung\covid_data0424\src/*',need_rotate=True,
                  need_resize=256,Hu_window=(-1000,512), need_tune=None):

    # nii_path=glob(nii_path)
    nii_path = [_nii_path for _nii_path in glob(nii_path) if _nii_path.endswith('nii') or _nii_path.endswith('nii.gz')]

    nii_path.sort()
    print('Finding ', len(nii_path),' nii.gz format files.\t')
    tag=1
    total_list = []

    for name in nii_path:
        nii = get_itk_array(name,False)
        print('Reading:\t', name.split('/')[-1])
        print(nii.shape)

        Hu_min = Hu_window[0]
        Hu_max = Hu_window[1]
        nii[nii<Hu_min] = Hu_min
        nii[nii>Hu_max] = Hu_max
        nii = nii - np.min(nii)
        nii = nii * 1.0 / np.max(nii)


        if len(nii.shape)>=4:
            nii = nii[:,:,:,0]


        slices = nii.shape[0]
        if need_rotate:
            for i in range(slices):
                nii[i, :, :] = np.flip(nii[i, :, :])
                nii[i, :, :] = np.flip(nii[i, :, :],axis=1)

        if need_resize:
            total_temp = np.zeros((slices, need_resize, need_resize))
            for i in range(slices):
                total_temp[i] = imresize(nii[i], (need_resize, need_resize))
        else:
            total_temp = nii

        tag = tag + 1
        total_list.append(total_temp)

    total = total_list.pop(0)
    for i in total_list:
        total = np.concatenate((total,i),0)

    total_all = total
    if np.max(total_all)>1:
        total_all = total_all - np.min(total_all)
        total_all = total_all * 1.0 / np.max(total_all)


    print('Done.')

    return total_all


def save_pred_to_nii(pred=None,save_path=r'E:\Lung\covid_data0424\label_V1pred/',ref_path=r'E:\Lung\covid_data0424\src/*',
                     need_rotate=True, need_resize= True, need_threshold=True):

    # nii_path=glob(ref_path)
    nii_path = [_nii_path for _nii_path in glob(ref_path) if _nii_path.endswith('nii') or _nii_path.endswith('nii.gz')]

    nii_path.sort()
    print('\n**********\t',len(nii_path),'file(s) to save:','\t**********\n')


    tag = 0
    for name in nii_path:
        nii = get_itk_array(name,False)
        nii_ref = get_itk_image(name)
        slices = nii.shape[0]
        matrix = nii.shape[1:]

        if tag + slices > pred.shape[0]:
            cut = tag + slices - pred.shape[0]
            H = pred.shape[1]  # 256
            W = pred.shape[2]  # 256
            temp = np.zeros((cut, H, W))
            pred = np.concatenate((pred, temp), 0)

        nii_one = pred[tag:tag+slices]

        if need_rotate:
            for i in range(slices):
                nii_one[i, :, :] = np.flip(nii_one[i, :, :])
                nii_one[i, :, :] = np.flip(nii_one[i, :, :], axis=1)


        if need_resize:
            total_temp = np.zeros((slices, matrix[0], matrix[1]))
            for i in range(slices):
                total_temp[i] = imresize(nii_one[i], (matrix[0], matrix[1]), interp='nearest')

        else:
            total_temp = nii_one

        if need_threshold:
            if total_temp.max()>=200:
                total_temp[total_temp<128]=0
                total_temp[total_temp>=128]=1

        print('Saving:\t',total_temp.shape)
        if '\\' in save_path:
            write_itk_imageArray(total_temp, save_path+name.split('\\')[-1], nii_ref)  # for Windows
        else:
            write_itk_imageArray(total_temp, save_path+name.split('/')[-1], nii_ref)  # for Linux

        tag = tag + slices

    print('Done.')

    return None


if __name__ == '__main__':
    pass
