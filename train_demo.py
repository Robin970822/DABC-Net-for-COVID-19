# -*- coding: utf-8 -*-
import numpy as np
import os
import time
from sklearn.model_selection import KFold

# from warnings import warn
"""
(optional)
set backend to avoid OOM error
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

'''
Parameter
'''
Slice_count = 6
Height = 256
Width = 256
Channel = 1  # for grayscale image
Batch_size = 1

'''
Make dataset
(Not sure customized data whether can be loaded by SimpleITK)

Shape of source and label data: (slices, Height, Width, channel). Type: numpy.ndarray
The slices need to be arranged in order(without shuffle).
'''
pass

'''
read all data from .npy
Got shape: (slices, 256, 256, 1)
'''
all_src_data = np.load('path-to-source-data.npy')
all_src_data = np.expand_dims(all_src_data, -1)
all_mask_data = np.load('path-to-label-data.npy')
all_mask_data = np.expand_dims(all_mask_data, -1)

if np.max(all_src_data) > 1:
    all_src_data = all_src_data / 255.0
    print('Normlised!\n')
if np.max(all_mask_data) > 1:
    all_mask_data = all_mask_data / 255.0
    print('Normlised!\n')
print('Data loaded !\n')

'''
data generator
'''
def gen_chunk(in_img, in_mask, slice_count = 2, batch_size = 16):
    while True:
        img_batch = []
        mask_batch = []
        for _ in range(batch_size):
            s_idx = np.random.choice(range(in_img.shape[0]-slice_count))
            img_batch += [in_img[s_idx:(s_idx+slice_count)]]
            mask_batch += [in_mask[s_idx:(s_idx+slice_count)]]
        yield np.stack(img_batch, 0), np.stack(mask_batch, 0)

'''
Augmentation(optional)
If empty, this procedure will skip
'''
from keras.preprocessing.image import ImageDataGenerator

d_gen = ImageDataGenerator(
                               # # rotation_range=45,
                               # width_shift_range=0.25,  # 0.15
                               # height_shift_range=0.25,
                               # # shear_range=0.1,
                               # zoom_range=0.2,  # 0.25
                               # fill_mode='nearest',
                               # # horizontal_flip=True,
                               # # vertical_flip=True,
                               # # featurewise_center =True,
                               # # featurewise_std_normalization = True,
                               # # samplewise_center=True,
                               # # samplewise_std_normalization= True,
                               # # zca_epsilon=1e-06,
                           )

def gen_aug_chunk(in_gen):
    for i, (x_img, y_img) in enumerate(in_gen):
        xy_block = np.concatenate([x_img, y_img], 1).swapaxes(1, -1)[:, 0]
        img_gen = d_gen.flow(xy_block, shuffle=True, seed=i, batch_size = x_img.shape[0])
        xy_scat = next(img_gen)
        xy_scat = np.expand_dims(xy_scat,1).swapaxes(1, -1)
        yield xy_scat[:, :xy_scat.shape[1]//2], xy_scat[:, xy_scat.shape[1]//2:]


'''
train
'''
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam,sgd
from pipeline.inference_pipeline import local_evaluate


def train_func(model_name_id, opt = 'Adam(lr = 1e-4)', _test_vol=None, _test_mask=None, init_weight=None):
    print('######\t model_name_id: '+model_name_id)
    print('using opt: '+opt)
    opt_name = eval(opt)
    model = models.DABC(input_size=(Slice_count, Height, Width, Channel), opt=opt_name)
    if init_weight:  # for transfer learning and fine-tune
        model = model.DABC(input_size=(Slice_count, Height, Width, Channel), opt=opt_name, load_weighted=init_weight)
        print('\t init weight has been loaded!')

    import time
    time_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    time_id = str(time_id)[-8:]  # '02131414'

    if not os.path.exists('weight'):
        os.makedirs('weight')
    weight_path = 'weight/' + model_name_id + time_id
    print('\n********\t Save_weight_path: ', weight_path, '\t********\n')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only = True)  # monitor='val_loss' or dice

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-6, patience=15, verbose=1, epsilon=1e-4, mode='min')

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=30)
    # save model
    callbacks_list = [
                    checkpoint,
                    early,
                    reduceLROnPlat

                     ]

    model.fit_generator(train_aug_gen,
                            epochs=1,  # 50
                            steps_per_epoch = 100,
                            validation_data = valid_gen,
                            validation_steps=50,
                            callbacks = callbacks_list,
                            shuffle=True  # defualt
                           )


    '''
    evaluate and write result to file
    '''
    local_evaluate(_test_vol, _test_mask, model, _slice_count=Slice_count, threshold_after_infer=0.5)  # _test_vol:(312, 256, 256, 1)

    del model  # release RAM


    return None


'''
k-fold cross validation
'''
kf = KFold(n_splits=5)  # Number of folds.

tag = 1  # fold number
for train, test in kf.split(all_src_data,all_mask_data):
    print('\n**********\ttraing on %d fold:\t**********\n' % tag)
    # print("\n**********\t k-fold shape：%s %s\t**********\n" % (train.shape, test.shape))
    print("\n**********\t{} in {}-fold cross validation\t**********\n".format(str(tag), str(5)))

    train_vol = all_src_data[train]
    train_mask = all_mask_data[train]
    test_vol = all_src_data[test]
    test_mask = all_mask_data[test]

    print('test_vol\t',test_vol.shape)
    print('test_mask\t',test_mask.shape)

    '''
    train 
    '''
    from pipeline.data_pipeline import confirm_data

    test_vol = confirm_data(test_vol)
    test_mask = confirm_data(test_mask)

    train_gen = gen_chunk(train_vol, train_mask, slice_count=Slice_count,
                          batch_size=2)  # limited to GPU RAM
    valid_gen = gen_chunk(test_vol, test_mask, slice_count=Slice_count, batch_size=2)

    train_aug_gen = gen_aug_chunk(train_gen)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from models import models
    train_func(model_name_id='define-your-model-name-here' + '_fold' + str(tag) + '_',
               opt='Adam(lr = 1e-4)', _test_vol=test_vol, _test_mask=test_mask)

    tag = tag+1
    time.sleep(5)

print('Done.')
