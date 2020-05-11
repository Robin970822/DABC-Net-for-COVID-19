import os
import config
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import prob2binary


def calculate_volume(raw, lung, lesion, meta):
    lung = prob2binary(lung)
    lesion = prob2binary(lesion)
    lesion = lesion * lung
    former_slice = 0
    res_list = []
    for index, row in meta.iterrows():
        slices = row['slice']
        spacing = eval(row['spacing'])
        origin_shape = eval(row['shape'])
        size_factor = (origin_shape[1] * origin_shape[2]
                       ) / (lung.shape[1] * lung.shape[2])
        voxel_size = spacing[0] * spacing[1] * spacing[2] * size_factor

        total_slice = lung.shape[0]
        current_slice = np.min([former_slice + slices, total_slice])
        lung_volume = np.sum(lung[former_slice:current_slice]) * voxel_size
        lesion_volume = np.sum(lesion[former_slice:current_slice]) * voxel_size
        lung_lesion_volume = np.sum(
            lesion[former_slice:current_slice] * lung[former_slice:current_slice]) * voxel_size
        weighted_lesion_volume = np.sum(
            lesion[former_slice:current_slice] * raw[former_slice:current_slice]) * voxel_size

        ratio = lesion_volume / lung_volume
        res_list.append({
            'lung': lung_volume,
            'lesion': lesion_volume,
            'ratio': ratio,
            'lung_lesion': lung_lesion_volume,
            'weighted_lesion': weighted_lesion_volume})
        former_slice = current_slice
    return res_list


total_data = pd.DataFrame()
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

    if len(lesion.shape) > 3:
        lesion = lesion.reshape([lesion.shape[0] * lesion.shape[1], 256, 256])

    assert len(lesion.shape) == 3
    assert len(lung.shape) == 3

    res_list = calculate_volume(raw_data, lung, lesion, meta)
    res_df = pd.DataFrame(res_list)
    new_meta = pd.concat([meta, res_df], axis=1)
    total_data = pd.concat([total_data, new_meta])

total_data.to_csv('total_data_severe.csv')
