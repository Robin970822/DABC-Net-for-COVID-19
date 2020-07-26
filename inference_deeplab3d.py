import os
import config
import argparse
import numpy as np
import models_baseline_deeplabV3 as M

from tqdm import tqdm
from evaluate_pipe import my_evaluate1

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='Lung')
args = parser.parse_args()

model = args.model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = 'weight/Lung_DeeplabV3_Only 20_Xcep_06172259/'
model = M.Deeplabv3(activation='sigmoid', backbone='xception')
model.load_weights(model_name)

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

    '''
    evaluate
    '''
    model_name_id = model_name

    # don't have label to evaluate
    # return:(slices,256,256,1)
    pred = my_evaluate1(test_vol, test_mask, model, model_name_id=model_name_id, mode=2)
    print('Pred shape: {}'.format(pred.shape))
    if args.model == 'Covid':
        np.save(os.path.join(config.lesion_root, '{}_pred_lesion_deeplab.npy'.format(patientID)), pred)
    elif args.model == 'Lung':
        np.save(os.path.join(config.lung_root, '{}_pred_lung_deeplab.npy'.format(patientID)), pred)
