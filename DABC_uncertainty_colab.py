import keras
import numpy as np
from models import models_dropblock as Model
import tensorflow as tf
from pipeline.inference_pipeline import local_inference
from pipeline.data_pipeline import read_from_nii, save_pred_to_nii, confirm_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def DABC_uncertainty(nii_filename='', save_filename='', sample_value=10, uc_chosen='Predictive'):
    save_path = save_filename
    nii_path = nii_filename  # for Colab
    '''
    Check input (one) filename!
    '''
    if uc_chosen == 'Predictive':
        uncertainty = 4
    elif uc_chosen == 'Aleatoric':  # Aleatoric
        uncertainty = 1
    elif uc_chosen == 'Epistemic':
        uncertainty = 2
    elif uc_chosen == 'Both':
        uncertainty = 0
    else:
        print('Please chosen correct uncertainty name!')
        print(uc_chosen)
        return None
    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1000, 512), need_rotate=True)
    all_src_data = np.expand_dims(all_src_data, -1)

    print('\n**********\tInferring CT scans:\t**********\n')
    test_vol = confirm_data(all_src_data)

    keras.backend.clear_session()

    name = 'weight/Covid_dropblock_05271037'
    model = Model.DABC(input_size=(4, 256, 256, 1),
                       load_weighted=name, is_trainable=True)

    samples = [sample_value]
    pred = []

    for i in range(samples[0]):
        pred.append(
            local_inference(test_vol, model, threshold_after_infer=0))

    pred = np.squeeze(np.array(pred))

    pred = np.expand_dims(pred, 1)
    p_hat = pred
    p_hat = p_hat - np.min(p_hat)
    p_hat = p_hat * 1.0 / np.max(p_hat)
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    # ref_path = nii_path

    if uncertainty == 0:

        save_pred_to_nii(pred=np.squeeze(aleatoric), save_path=save_path + '\\aleatoric_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
        save_pred_to_nii(pred=np.squeeze(epistemic), save_path=save_path + '\\epistemic_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty == 1:
        save_pred_to_nii(pred=np.squeeze(aleatoric), save_path=save_path + '\\aleatoric_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty == 2:
        save_pred_to_nii(pred=np.squeeze(epistemic), save_path=save_path + '\\epistemic_', ref_path=nii_path,
                         need_resize=True, need_rotate=True, need_threshold=False)
    elif uncertainty == 4:
        save_pred_to_nii(pred=np.squeeze(epistemic + aleatoric), save_path=save_path + '\\predictive_',
                         ref_path=nii_path,
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
        DABC_uncertainty(nii_filename=args.input, save_filename=args.output)
