import os
import time
import config
import numpy as np
import models3_V86 as M

from tqdm import tqdm
from evaluate_performance_mouse_pipe import my_evaluate1

tag = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = 'Covid_models3_V86'
model = M.BCDU_net_D3(input_size=(4, 256, 256, 1), load_weighted=model_name)

for name in tqdm(config.npy_path):
    patientID = os.path.basename(name).split('.')[0] 
    raw_data = np.load(name)
    if len(raw_data.shape) == 3:
        raw_data = raw_data[..., np.newaxis]
    print('Raw Data shape: {}'.format(raw_data.shape))
    # train_vol = all_src_data[train]  # shape: (436, 160, 160, 1) 
    # train_mask = all_mask_data[train]  # (436, 160, 160, 1)
    test_vol = raw_data  # (110, 160, 160, 1)
    test_mask = raw_data  

    if test_vol.shape[0] % 4 != 0:  
        cut = test_vol.shape[0] % 4
        extend = 4 - cut
        extend_vol = np.zeros([extend, test_vol.shape[1], test_vol.shape[2], test_vol.shape[3]])
        test_vol = np.concatenate((test_vol, extend_vol), axis=0)
        # test_mask = np.concatenate((test_mask, extend_vol), axis=0)
    assert test_vol.shape[0] % 4 == 0

    '''
    evaluate
    '''
    time_id = np.int64(time.strftime(
        '%Y%m%d%H%M', time.localtime(time.time())))
    time_id = str(time_id)[-8:]  
    model_name_id = '6mm multi_' + model_name + '_infer_fold:' + str(tag)

    # don't have label to evaluate
    # return:(slices,256,256,1)
    pred = my_evaluate1(test_vol, test_mask, model,
                        model_name_id=model_name_id)
    pred_path = os.path.join(config.lesion_root, '{}_pred_lesion.npy'.format(patientID))
    print('Pred shape: {} in {}'.format(pred.shape, pred_path))
    np.save(pred_path, pred)

