<<<<<<< HEAD
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_all_data_from_nii_pipe import read_from_nii


def infer_colab(nii_path='', save_path='', sform_code=0):
    # save_path = save_path + '\\*'
    # nii_path = nii_path + '\\*'  # for Windows
    save_path = save_path + '/*'
    nii_path = nii_path + '/*'  # for colab

    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1024, 512), need_rotate=True)

    all_src_data = np.expand_dims(all_src_data, -1)

    all_mask_data = np.zeros_like(all_src_data)

    '''

    '''
    from evaluate_performance_pipeline import my_evaluate

    print('\n**********\tInferring CT scans:\t**********\n')

    test_vol = all_src_data
    test_mask = all_mask_data

    if test_vol.shape[0] % 4 != 0:
        cut = test_vol.shape[0] % 4
        test_vol = test_vol[:-cut]
        test_mask = test_mask[:-cut]
    assert test_vol.shape[0] % 4 == 0

    '''
    infer
    '''
    import models as Model

    name = 'weight/Covid_05112327'
    model = Model.DABC(input_size=(4, 256, 256, 1),
                          load_weighted=name)


    pred = my_evaluate(test_vol, test_mask, model, model_name_id=None, only_infer=True, )  # (slices,H,W,1)已经处理去掉了最后一维


    from read_all_data_from_nii_pipe import save_pred_to_nii

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
=======
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_all_data_from_nii_pipe import read_from_nii


def infer_colab(nii_path='', save_path='', sform_code=0):
    # save_path = save_path + '\\*'
    # nii_path = nii_path + '\\*'  # for Windows
    save_path = save_path + '/*'
    nii_path = nii_path + '/*'  # for colab

    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1024, 512), need_rotate=True)

    all_src_data = np.expand_dims(all_src_data, -1)

    all_mask_data = np.zeros_like(all_src_data)

    '''

    '''
    from evaluate_performance_pipeline import my_evaluate

    print('\n**********\tInferring CT scans:\t**********\n')

    test_vol = all_src_data
    test_mask = all_mask_data

    if test_vol.shape[0] % 4 != 0:
        cut = test_vol.shape[0] % 4
        test_vol = test_vol[:-cut]
        test_mask = test_mask[:-cut]
    assert test_vol.shape[0] % 4 == 0

    '''
    infer
    '''
    import models as Model

    name = 'weight/Covid_05112327'
    model = Model.DABC(input_size=(4, 256, 256, 1),
                          load_weighted=name)


    pred = my_evaluate(test_vol, test_mask, model, model_name_id=None, only_infer=True, )  # (slices,H,W,1)已经处理去掉了最后一维


    from read_all_data_from_nii_pipe import save_pred_to_nii

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
>>>>>>> 895a89b8f00b3828e490646c89e56b4aa42483e2
