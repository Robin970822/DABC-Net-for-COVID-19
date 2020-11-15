# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd


def get_days(datetime):
    return datetime.days


def get_sex(patient_sex):
    return 1 if patient_sex == 'M' else 0


def min_max_scalar(x, min, max):
    return (x - min) / (max - min)


def crop_volume(volume, crop):
    volume_ = volume.copy()
    volume_[:crop[0]] = 0
    volume_[-crop[1]:] = 0
    return volume_


def prob2binary(prob, thresh=0.5):
    res = np.zeros_like(prob)
    res[prob > thresh] = 1
    return res


def resize(data, shape):
    mask = np.zeros(shape)
    for i in range(shape[0]):
        mask[i, :, :] = cv2.resize(data[i, :, :], (shape[1], shape[2]))
    return mask


def get_z(lesion):
    z = 0
    for i in range(lesion.shape[0]):
        z = z + i * np.sum(lesion[i, :, :])
    return z / (np.sum(lesion) + 1e-5)


def get_left_right(data, mid):
    right = data[:, :, :mid]
    left = data[:, :, mid:mid * 2]
    return left, right


def get_consolidation(raw, lung, lesion, thresh=0.5):
    """ Consolidation from raw, lung and lesion
    """
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

    lung_lesion_union_open_area = raw * lung_lesion_union_open[:raw.shape[0]]

    _, thres_image = cv2.threshold(
        lung_lesion_union_open_area, thresh, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thres_image_open = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN, kernel)

    return thres_image_open


def calculate_volume(raw, lung, lesion, meta, crop=None):
    if crop is None:
        crop = [0.0, 0.0]
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
        left_lesion, right_lesion = get_left_right(lung_lesion_current, mid)
        left_raw, right_raw = get_left_right(raw_current, mid)

        consolidation = get_consolidation(
            raw_current, lung_current, lesion_current)
        lesion_consolidation = lung_lesion_current * consolidation
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

        z = get_z(lung_lesion_current)
        left_z = get_z(left_lesion)
        right_z = get_z(right_lesion)

        ratio = lung_lesion_volume / lung_volume
        left_ratio = left_lesion_volume / left_lung_volume
        right_ratio = right_lesion_volume / right_lung_volume
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
                'left_ratio': left_ratio,
                'right_ratio': right_ratio,

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


def calculate(raw, lung, lesion, meta):
    meta = meta[meta['slice'] > 300]  # select thin scans
    meta = meta.reset_index()  # DataFrame index reset

    res_list = calculate_volume(raw, lung, lesion, meta, crop=[0.17, 0.08])
    all_info = pd.concat([meta, pd.DataFrame(res_list)], axis=1)
    return res_list, all_info


def preprocessing(all_info, feature):
    # transfer PatientSex into 0/1
    all_info['sex'] = all_info['PatientSex'].map(get_sex)
    # normalize the z-position by dividing the slice number of the CT scan
    all_info['z'] = all_info['z'] / all_info['slice']
    all_info['left_z'] = all_info['left_z'] / all_info['slice']
    all_info['right_z'] = all_info['right_z'] / all_info['slice']

    X = all_info[feature].astype(np.float32)
    return X
