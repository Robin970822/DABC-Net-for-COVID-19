import os
import config
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import prob2binary, crop_volume


def calculate_uncertain(lesion, meta, aleatoric, epistemic):
    lesion = prob2binary(lesion)
    former_slice = 0
    res_list = []
    for index, row in meta.iterrows():
        slices = row['slice']
        total_slice, height, width = lesion.shape
        current_slice = np.min([former_slice + slices, total_slice])
        if former_slice == current_slice:
            continue

        lesion_current = lesion[former_slice:current_slice]
        aleatoric_current = aleatoric[former_slice:current_slice]
        epistemic_current = epistemic[former_slice:current_slice]

        aleatoric_max = np.max(aleatoric_current)
        aleatoric_mean = np.mean(aleatoric_current)
        aleatoric_ave = np.sum(aleatoric_current) / (np.sum(lesion_current) + 1e-5)

        epistemic_max = np.max(epistemic_current)
        epistemic_mean = np.mean(epistemic_current)
        epistemic_ave = np.sum(epistemic_current) / (np.sum(lesion_current) + 1e-5)

        res_list.append({
            'aleatoric_max': aleatoric_max,
            'aleatoric_mean': aleatoric_mean,
            'aleatoric_ave': aleatoric_ave,
            'epistemic_max': epistemic_max,
            'epistemic_mean': epistemic_mean,
            'epistemic_ave': epistemic_ave
        })

        former_slice = current_slice
    return res_list


if __name__ == '__main__':
    total_data = pd.DataFrame()
    for name in tqdm(config.npy_path):
        patientID = os.path.basename(name).split('.')[0]
        lesion_path = os.path.join(
            config.uncertain_root, '{}_pred.npy'.format(patientID))
        meta_path = os.path.join(config.meta_root, '{}.csv'.format(patientID))
        aleatoric_path = os.path.join(
            config.uncertain_root, '{}_aleatoric.npy'.format(patientID))
        epistemic_path = os.path.join(
            config.uncertain_root, '{}_epistemic.npy'.format(patientID))

        lesion = np.load(lesion_path)
        lesion = np.mean(lesion, axis=0)
        aleatoric = np.load(aleatoric_path)
        epistemic = np.load(epistemic_path)
        meta = pd.read_csv(meta_path, index_col=[0])

        print(lesion.shape)

        # lung = remove_small(lung, slices=lung.shape[0], min_size=16)
        if config.rotate == '_rotate':
            lesion = np.flip(lesion, axis=1)
            aleatoric = np.flip(aleatoric, axis=1)
            epistemic = np.flip(epistemic, axis=1)

        res_list = calculate_uncertain(
            lesion, meta, aleatoric, epistemic)
        res_df = pd.DataFrame(res_list)
        new_meta = pd.concat([meta, res_df], axis=1)
        total_data = pd.concat([total_data, new_meta])

    total_data.to_csv(config.uncertain_csv_name)
