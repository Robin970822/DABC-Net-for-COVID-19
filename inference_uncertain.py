import os
import time
import config
import numpy as np
import models3_V86_xy as M

from tqdm import tqdm
from evaluate_pipe import my_evaluate1

sample_times = config.sample_times

name = 'weight/Covid_V86_7th_manual_epoch50_No Aug_fold3_05270106'  # covid
model = M.BCDU_net_D3(input_size=(4, 256, 256, 1),
                      load_weighted=name, is_trainable=True)

for name in tqdm(config.npy_path):
    patientID = os.path.basename(name).split('.')[0]
    raw_data = np.load(name)
    if len(raw_data.shape) == 3:
        raw_data = raw_data[..., np.newaxis]
    print('Raw Data shape: {}'.format(raw_data.shape))
    # train_vol = all_src_data[train]  # shape: (436, 160, 160, 1) 5折/80%
    # train_mask = all_mask_data[train]  # (436, 160, 160, 1)
    test_vol = raw_data  # (110, 160, 160, 1)
    test_mask = raw_data  # 这里暂时没label，用test占位

    # todo 这里可能需要修改一下，slices需要4的倍数，之前图方便直接去掉。主要会影响最后一个nii，计算指标的话可忽略。可用全零图像补全。
    if test_vol.shape[0] % 4 != 0:  # 若不被4整除
        cut = test_vol.shape[0] % 4
        extend = 4 - cut
        extend_vol = np.zeros(
            [extend, test_vol.shape[1], test_vol.shape[2], test_vol.shape[3]])
        test_vol = np.concatenate((test_vol, extend_vol), axis=0)
        test_mask = np.concatenate((test_mask, extend_vol), axis=0)
    assert test_vol.shape[0] % 4 == 0

    model_name_id = 'infer_' + name
    pred = []

    for i in range(sample_times):
        pred.append(my_evaluate1(test_vol, test_mask, model, model_name_id=model_name_id))

    pred = np.squeeze(np.array(pred))
    pred = np.expand_dims(pred, 1)
    p_hat = pred
    prediction = np.mean(p_hat, axis=0)
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    print('Pred shape: {}'.format(np.squeeze(pred).shape))
    print('aleatoric shape: {}'.format(np.squeeze(aleatoric).shape))
    print('epistemic shape: {}'.format(np.squeeze(epistemic).shape))
    np.save(os.path.join(config.uncerten_root, '{}_pred.npy'.format(patientID)), np.squeeze(pred))
    np.save(os.path.join(config.uncerten_root, '{}_aleatoric.npy'.format(patientID)), np.squeeze(aleatoric))
    np.save(os.path.join(config.uncerten_root, '{}_epistemic.npy'.format(patientID)), np.squeeze(epistemic))
