import numpy as np
import tensorflow as tf
from models import models as Model
from pipeline.inference_pipeline import local_inference
from pipeline.data_pipeline import save_pred_to_nii, read_from_nii, confirm_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def infer_colab(nii_path='', save_path='', usage='covid'):
    save_path = save_path + '/*'
    nii_path = nii_path + '/*'
    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1024, 512), need_rotate=True)
    all_src_data = np.expand_dims(all_src_data, -1)

    print('\n**********\tInferring CT scans:\t**********\n')
    test_vol = confirm_data(all_src_data)
    '''
    infer
    '''
    if usage == 'covid':
        name = 'weight/Covid_05112327'
    elif usage == 'lung':
        name = 'weight/model_05090017'
    else:
        print('Please select correct model!')
        return None
    model = Model.DABC(input_size=(4, 256, 256, 1), load_weighted=name)
    pred = local_inference(test_vol, model)
    save_pred_to_nii(pred=pred, save_path=save_path.replace('*', ''), ref_path=nii_path,
                     need_resize=True, need_rotate=True)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='input', required=True, type=str, help="Input path")
    parser.add_argument("-o", dest='output', required=True, type=str, help="Output path")
    parser.add_argument("-u", dest='uncertainty', help="Uncertainty")
    args = parser.parse_args()

    if not os.path.exists(args.input) or not os.path.exists(args.output):
        print('\nThe path does not exist.\n')
    elif 'gz' not in os.listdir(args.input)[0]:
        print('\nThe path does not contain nii.gz format files.\n')
    else:
        infer_colab(nii_path=args.input, save_path=args.output)
