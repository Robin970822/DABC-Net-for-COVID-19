import os
import time
import config
import argparse
import numpy as np
import models as M

from tqdm import tqdm
from evaluate_pipe import my_evaluate1

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='Lung')
args = parser.parse_args()

model = args.model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = '{}_models3_V86_all_8_tf12_gobackFalse_04230307_fold3'.format(model)
model = M.BCDU_net_D3(input_size=(4, 256, 256, 1), load_weighted=model_name)


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
        extend_vol = np.zeros([extend, test_vol.shape[1], test_vol.shape[2], test_vol.shape[3]])
        test_vol = np.concatenate((test_vol, extend_vol), axis=0)
        test_mask = np.concatenate((test_mask, extend_vol), axis=0)
    assert test_vol.shape[0] % 4 == 0

    '''
    evaluate
    '''
    model_name_id = '6mm multi_' + model_name

    # don't have label to evaluate
    # return:(slices,256,256,1)
    pred = my_evaluate1(test_vol, test_mask, model, model_name_id=model_name_id)
    print('Pred shape: {}'.format(pred.shape))
    if args.model == 'Covid':
        np.save(os.path.join(config.lesion_root, '{}_pred_lesion.npy'.format(patientID)), pred)
    elif args.model == 'Lung':
        np.save(os.path.join(config.lung_root, '{}_pred_lung.npy'.format(patientID)), pred)
