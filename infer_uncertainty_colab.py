import keras
import numpy as np
import tensorflow as tf

import models.models_dropblock as Model
from utils.evaluate_performance_pipeline import local_evaluate
from utils.read_all_data_from_nii_pipe import read_from_nii

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def infer_uncertainty(nii_filename='', save_filename='', sample_value=10,uc_chosen='Predictive',
                      threshold_value=0.5, sform=None,
                      uncertainty=0,
                      ):
    save_path = save_filename
    nii_path = nii_filename  # for Colab


    '''
    Check input (one) filename!
    '''
    if uc_chosen=='Predictive':
        uncertainty=4
    elif uc_chosen=='Aleatoric':  # Aleatoric
        uncertainty=1
    elif uc_chosen=='Epistemic':
        uncertainty=2
    elif uc_chosen=='Both':
        uncertainty=0
    else:
        print('Please chosen correct uncertainty name!')
        print(uc_chosen)
        return None

    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1000, 512), need_rotate=True)

    all_src_data = np.expand_dims(all_src_data, -1)

    all_mask_data = np.zeros_like(all_src_data)

    '''

    '''

    print('\n**********\tInferring CT scans:\t**********\n')

    test_vol = all_src_data
    test_mask = np.zeros_like(all_mask_data)

    if test_vol.shape[0]%4 != 0:
        cut = test_vol.shape[0] % 4
        test_vol = test_vol[:-cut]
        test_mask = test_mask[:-cut]
    assert test_vol.shape[0]%4 == 0


    keras.backend.clear_session()

    name = 'weight/Covid_dropblock_05271037'
    model = Model.DABC(input_size=(4, 256, 256, 1),
                          load_weighted=name, is_trainable=True)


    samples=[sample_value]
    pred = []

    for i in range(samples[0]):
        pred.append(local_evaluate(test_vol, test_mask, model, model_name_id=None, only_infer=True, threshold_after_infer=0))

    pred = np.squeeze(np.array(pred))

    from utils.read_all_data_from_nii_pipe import save_pred_to_nii

    pred = np.expand_dims(pred, 1)
    p_hat = pred
    p_hat = p_hat - np.min(p_hat)
    p_hat = p_hat * 1.0 / np.max(p_hat)
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    # ref_path = nii_path

    if uncertainty==0:

        save_pred_to_nii(pred=np.squeeze(aleatoric), save_path=save_path + '\\aleatoric_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
        save_pred_to_nii(pred=np.squeeze(epistemic), save_path=save_path + '\\epistemic_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty==1:
        save_pred_to_nii(pred=np.squeeze(aleatoric), save_path=save_path + '\\aleatoric_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty==2:
        save_pred_to_nii(pred=np.squeeze(epistemic), save_path=save_path + '\\epistemic_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty==4:
        save_pred_to_nii(pred=np.squeeze(epistemic+aleatoric), save_path=save_path + '\\predictive_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='input', required=True, type=str, help="Input filename")
    parser.add_argument("-o", dest='output', required=True, type=str, help="Output filename")
    parser.add_argument("-u", dest='uncertainty', type=int, choices=[0, 1, 2], default=0, help="Uncertainty")
    args = parser.parse_args()

    if not os.path.exists(args.input) or not os.path.exists(args.output):
        print('\nThe path does not exist.\n')
    elif 'gz' not in args.input:
        print('\nThe path does not contain nii.gz format files.\n')
    else:
        infer_uncertainty(nii_filename=args.input, save_filename=args.output, uncertainty=args.uncertainty)
