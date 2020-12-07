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
Slice_count = 8
Height = 256
Width = 256
Channel = 1  # for grayscale image
Batch_size = 2

Patch_height = Height
Patch_width = Width
Patch_slice = Slice_count
'''
Make dataset
(Not sure customized data whether can be loaded by SimpleITK)

Shape of each source and label data: (slices, Height, Width, channel). Type: List object consists of numpy.ndarray
The slices need to be arranged in order(without shuffle).
'''
pass

'''
Read all data from raw data. (In this demo, raw data is in nii.gz format)
Got : [(slices, 256, 256, 1), (slices, 256, 256, 1), ... ]
'''
from pipeline.data_pipeline import read_from_nii2list

all_src_data = read_from_nii2list(r'E:\Lung\covid_data0424\thick_src\2020035473*', need_resize=False, need_rotate=True)

all_mask_data = read_from_nii2list(r'E:\Lung\covid_data0424\thick_lung_label_V0\2020035473*', need_resize=False, need_rotate=True)

print('Data loaded !\n')


'''
pre-process
'''
from utils.pre_processing import my_PreProc

for i,data in enumerate(all_src_data):  # data:(396,256,256)
    data = np.expand_dims(data,-1)
    data = np.swapaxes(data, 1, 3)  # got (396, 1, 256, 256)
    data = my_PreProc(data)
    data = np.swapaxes(data, 1, 3)  # got (396, 256, 256, 1)
    all_src_data[i] = data

for i,data in enumerate(all_mask_data):  # data:(396,256,256)
    all_mask_data[i] = np.expand_dims(data,-1)

'''
get patches
'''


from utils.extract_patches import get_data_training
N_subimgs = 190  # patches obtain from each scan.
total_patches_imgs_train = np.zeros((N_subimgs*len(all_src_data), 1, Patch_height, Patch_width, Patch_slice))  # (190*3, 1, 128, 128, 4)
total_patches_masks_train = np.zeros((N_subimgs*len(all_mask_data), 1, Patch_height, Patch_width, Patch_slice))

for scan_id in range(len(all_src_data)):
    patches_imgs_train, patches_masks_train = get_data_training(
        train_imgs_original=all_src_data.pop(0),
        train_groudTruth=all_mask_data.pop(0),
        patch_height=Patch_height,  # 48
        patch_width=Patch_width,  # 48
        patch_slice=Patch_slice,
        N_subimgs=N_subimgs,  # Not fixed in train process. N_subimgs= num_RawDate(=1) * patches-per-data
        inside_FOV=False
    )
    total_patches_imgs_train[scan_id*N_subimgs:scan_id*N_subimgs+N_subimgs] = patches_imgs_train
    total_patches_masks_train[scan_id*N_subimgs:scan_id*N_subimgs+N_subimgs] = patches_masks_train


print('total_patches_imgs_train shape:',total_patches_imgs_train.shape)
print('total_patches_masks_train shape:',total_patches_masks_train.shape)

'''
data generator using keras
'''
import keras
batch_size = Batch_size


class gen(keras.utils.Sequence):
    def __init__(self, batch_size, train_data, mask_data):
        self.batch_size = batch_size
        self.train_data = train_data
        self.mask_data = mask_data

    def __len__(self):
        return len(total_patches_imgs_train) // self.batch_size

    def __getitem__(self, idx):
        # total_patches_imgs_train.shape (570, 1, 128, 128, 4)
        i = idx * self.batch_size
        x_batch = self.train_data[i : i + self.batch_size]
        x_batch = np.swapaxes(x_batch, 1, 4)  # (2(batch_size), 4, 256, 256, 1)

        y_batch = self.mask_data[i : i + self.batch_size]
        y_batch = np.swapaxes(y_batch, 1, 4)
        return x_batch,y_batch


'''
train
'''
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam,sgd
# from pipeline.inference_pipeline import local_evaluate


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

    # model.fit(train_gen,
    #         epochs=500,
    #         steps_per_epoch = 100,
    #         validation_data = valid_gen,
    #         validation_steps=50,
    #         callbacks = callbacks_list,
    #         shuffle=False
    #                        )

    model.fit_generator(train_gen,
            epochs=500,
            # steps_per_epoch = 100,
            validation_data = valid_gen,
            validation_steps=50,
            callbacks = callbacks_list,
            shuffle=False
                           )
    '''
    evaluate and write result to file
    '''
    # local_evaluate(_test_vol, _test_mask, model, _slice_count=Slice_count, threshold_after_infer=0.5)  # _test_vol:(312, 256, 256, 1)

    del model  # release RAM


    return None


'''
k-fold cross validation
'''
kf = KFold(n_splits=5)  # Number of folds.

tag = 1  # fold number
for train, test in kf.split(total_patches_imgs_train, total_patches_imgs_train):
    print('\n**********\ttraing on %d fold:\t**********\n' % tag)
    # print("\n**********\t k-fold shape：%s %s\t**********\n" % (train.shape, test.shape))
    print("\n**********\t{} in {}-fold cross validation\t**********\n".format(str(tag), str(5)))

    train_gen = gen(batch_size, train_data=total_patches_imgs_train[train], mask_data=total_patches_masks_train[train])
    valid_gen = gen(batch_size, train_data=total_patches_imgs_train[test], mask_data=total_patches_masks_train[test])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from models import models
    train_func(model_name_id='define-your-model-name-here' + '_fold' + str(tag) + '_',
               opt='Adam(lr = 1e-5)',
               # _test_vol=test_vol, _test_mask=test_mask
               )

    tag = tag+1
    time.sleep(5)

print('Done.')
