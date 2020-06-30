import os
import config
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import write_itk_imageArray, crop_volume
from scipy.misc.pilutil import imresize


def resize(data, shape):
    mask = np.zeros(shape)
    for i in range(shape[0]):
        mask[i, :, :] = imresize(data[i, :, :], (shape[1], shape[2]))
    mask[mask > 0] = 1
    return mask


def save_nii(lung, lesion, meta, lung_root, lesion_root, crop=[0, 0]):
    former_slice = 0
    for index, row in meta.iterrows():
        filename = row['filename']
        filename = os.path.basename(filename)
        slices = row['slice']
        origin_shape = eval(row['shape'])
        total_slice = lung.shape[0]
        current_slice = np.min([former_slice + slices, total_slice])

        lung_volume = lung[former_slice:current_slice]
        lesion_volume = lesion[former_slice:current_slice]

        if crop[0] > 0:
            lung_volume = crop_volume(lung_volume, (np.array(crop) * slices).astype('int'))
            lesion_volume = crop_volume(lesion_volume, (np.array(crop) * slices).astype('int'))

        lung_volume = resize(lung_volume, origin_shape)
        lesion_volume = resize(lesion_volume, origin_shape)

        lung_path = os.path.join(lung_root, filename)
        lesion_path = os.path.join(lesion_root, filename)

        write_itk_imageArray(lung_volume, lung_path)
        write_itk_imageArray(lesion_volume, lesion_path)
        former_slice = current_slice


if __name__ == '__main__':
    for name in tqdm(config.npy_path):
        patientID = os.path.basename(name).split('.')[0]
        raw_path = os.path.join(config.raw_root, '{}.npy'.format(patientID))
        lung_path = os.path.join(
            config.lung_root, '{}_pred_lung.npy'.format(patientID))
        lesion_path = os.path.join(
            config.lesion_root, '{}_pred_lesion.npy'.format(patientID))
        meta_path = os.path.join(config.meta_root, '{}.csv'.format(patientID))
        raw_data = np.load(raw_path)
        lung = np.load(lung_path)
        lesion = np.load(lesion_path)
        meta = pd.read_csv(meta_path, index_col=[0])
        # lung = remove_small(lung, slices=lung.shape[0])
        if config.rotate == '_rotate':
            lung = np.flip(lung, axis=1)
            lesion = np.flip(lesion, axis=1)
        save_nii(lung, lesion, meta, config.lung_root, config.lesion_root, crop=[0.17, 0.08])
