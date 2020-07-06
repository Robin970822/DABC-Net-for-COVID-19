import os
import cv2
import config
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import prob2binary, crop_volume
from skimage import morphology


def remove_small(data, slices, min_size=64):
    new_data = np.zeros_like(data)
    for i in range(slices):
        new_data[i, :, :] = morphology.remove_small_objects(data[i, :, :].astype(np.int), min_size=int(
            np.sin(i / slices * 3.14) * min_size), connectivity=8, in_place=True)
    return new_data


def get_z(lesion):
    z = 0
    for i in range(lesion.shape[0]):
        z = z + i * np.sum(lesion[i, :, :])
    return z / np.sum(lesion)


def get_left_right(data, mid):
    right = data[:, :, :mid]
    left = data[:, :, mid:mid * 2]
    return left, right


def get_consolidation(raw_data, lung, lesion, thres=0.5):
    lung = prob2binary(lung)
    lesion = prob2binary(lesion)

    lung_lesion_union = lesion.astype(np.uint8) | lung.astype(np.uint8)

    lung_lesion_union = lung_lesion_union.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lung_lesion_union_close = cv2.morphologyEx(
        lung_lesion_union, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lung_lesion_union_open = cv2.morphologyEx(
        lung_lesion_union_close, cv2.MORPH_CLOSE, kernel)

    lung_lesion_union_open_area = raw_data * \
        lung_lesion_union_open[:raw_data.shape[0]]

    _, thres_image = cv2.threshold(
        lung_lesion_union_open_area, thres, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thres_image_open = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN, kernel)

    return thres_image_open


def calculate_volume(raw, lung, lesion, meta, crop=[0, 0]):
    lung = prob2binary(lung)
    lesion = prob2binary(lesion)
    lung_lesion = lesion * lung
    former_slice = 0
    res_list = []
    for index, row in meta.iterrows():
        slices = row['slice']
        spacing = eval(row['spacing'])
        origin_shape = eval(row['shape'])
        size_factor = (origin_shape[1] * origin_shape[2]
                       ) / (lung.shape[1] * lung.shape[2])
        voxel_size = spacing[0] * spacing[1] * spacing[2] * size_factor

        total_slice, height, width = lung.shape
        mid = int(width / 2)
        current_slice = np.min([former_slice + slices, total_slice])

        lung_current = lung[former_slice:current_slice]
        lesion_current = lesion[former_slice:current_slice]
        lung_lesion_current = lung_lesion[former_slice:current_slice]
        raw_current = raw[former_slice:current_slice]
        if crop[0] > 0:
            lung_current = crop_volume(
                lung_current, (np.array(crop) * slices).astype('int'))
            lesion_current = crop_volume(
                lesion_current, (np.array(crop) * slices).astype('int'))
            lung_lesion_current = crop_volume(
                lung_lesion_current, (np.array(crop) * slices).astype('int'))

        left_lung, right_lung = get_left_right(lung_current, mid)
        left_lesion, right_lesion = get_left_right(lesion_current, mid)
        left_raw, right_raw = get_left_right(raw_current, mid)

        consolidation = get_consolidation(
            raw_current, lung_current, lesion_current)
        lesion_consolidation = lesion_current * consolidation
        left_consolidation, right_consolidation = get_left_right(
            lesion_consolidation, mid)

        calculate_list = [
            lung_current, lesion_current, lung_lesion_current,
            left_lung, right_lung, left_lesion, right_lesion,

            lesion_current * raw_current, lung_lesion_current * raw_current,
            left_lesion * left_raw, right_lesion * right_raw,

            consolidation, lesion_consolidation,
            left_consolidation, right_consolidation
        ]

        [
            lung_volume, lesion_volume, lung_lesion_volume,
            left_lung_volume, right_lung_volume, left_lesion_volume, right_lesion_volume,

            weighted_lesion_volume, weighted_lung_lesion_volume,
            left_weighted_lesion_volume, right_weighted_lesion_volume,

            consolidation_volume, lesion_consolidation_volume,
            left_consolidation_volume, right_consolidation_volume
        ] = map(lambda x: np.sum(x) * voxel_size, calculate_list)

        z = get_z(lesion_current)
        left_z = get_z(left_lesion)
        right_z = get_z(right_lesion)

        ratio = lesion_volume / lung_volume
        res_list.append(
            {
                'lung': lung_volume,
                'lesion': lesion_volume,
                'ratio': ratio,
                'lung_lesion': lung_lesion_volume,
                'left_lung': left_lung_volume,
                'right_lung': right_lung_volume,
                'left_lesion': left_lesion_volume,
                'right_lesion': right_lesion_volume,

                'weighted_lesion': weighted_lesion_volume,
                'weighted_lung_lesion': weighted_lung_lesion_volume,
                'left_weighted_lesion': left_weighted_lesion_volume,
                'right_weighted_lesion': right_weighted_lesion_volume,

                'consolidation': consolidation_volume,
                'lesion_consolidation': lesion_consolidation_volume,
                'left_consolidation': left_consolidation_volume,
                'right_consolidation': right_consolidation_volume,

                'z': z,
                'left_z': left_z,
                'right_z': right_z,
            }
        )
        former_slice = current_slice
    return res_list


if __name__ == '__main__':
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
            lesion = lesion.reshape(
                [lesion.shape[0] * lesion.shape[1], 256, 256])

        assert len(lesion.shape) == 3
        assert len(lung.shape) == 3

        # lung = remove_small(lung, slices=lung.shape[0], min_size=16)
        if config.rotate == '_rotate':
            lung = np.flip(lung, axis=1)
            lesion = np.flip(lesion, axis=1)
            # raw_data = np.flip(raw_data, axis=1)
        res_list = calculate_volume(
            raw_data, lung, lesion, meta, crop=[0.17, 0.08])
        res_df = pd.DataFrame(res_list)
        new_meta = pd.concat([meta, res_df], axis=1)
        total_data = pd.concat([total_data, new_meta])

    total_data.to_csv('lr_{}'.format(config.csv_name))
