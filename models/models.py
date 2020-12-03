from __future__ import division
from keras.models import Model
from keras.optimizers import *
from keras.layers import *
from .loss import dice_coef, weighted_dice_with_CE
from .block import resconv, slice_at_block

smooth = 0.001


def DABC(input_size=(10, 256, 256, 1), opt=Adam(lr=1e-4), load_weighted=None, ):
    _slice_count = input_size[0]
    droprate = 0.5
    is_trainable = False

    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                                   input_shape=(_slice_count, None, None, 1)))(inputs)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)

    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = resconv(pool1, 128, name='res_block1', is_batchnorm=True)

    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = resconv(pool2, 256, name='res_block2', is_batchnorm=True)

    drop3 = TimeDistributed(Dropout(droprate))(conv3, training=is_trainable)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3)
    conv4_1 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
    drop4_1 = TimeDistributed(Dropout(droprate))(conv4_1, training=is_trainable)
    conv4_2 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        drop4_1)
    conv4_2 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv4_2)
    conv4_2 = TimeDistributed(Dropout(droprate))(conv4_2, training=is_trainable)
    merge_dense = concatenate([conv4_2, drop4_1], axis=-1)
    conv4_3 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        merge_dense)
    conv4_3 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv4_3)
    drop4_3 = TimeDistributed(Dropout(droprate))(conv4_3, training=is_trainable)

    up6 = TimeDistributed(
        Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(drop4_3)
    up6 = TimeDistributed(BatchNormalization(axis=3))(up6)
    up6 = TimeDistributed(Activation('relu'))(up6)

    merge6 = concatenate([drop3, up6], axis=-1)

    conv6 = resconv(merge6, 256, name='resblock3')
    conv6 = slice_at_block(conv6, 512 // 2, _slice_count,name='DABC_1')

    up7 = TimeDistributed(
        Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv6)
    up7 = TimeDistributed(BatchNormalization(axis=3))(up7)
    up7 = TimeDistributed(Activation('relu'))(up7)

    merge7 = concatenate([conv2, up7], axis=-1)
    conv7 = resconv(merge7, 128, name='resblock4')
    conv7 = slice_at_block(conv7, 256 // 2, _slice_count, name='DABC_2')

    up8 = TimeDistributed(
        Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv7)
    up8 = TimeDistributed(BatchNormalization(axis=3))(up8)
    up8 = TimeDistributed(Activation('relu'))(up8)

    merge8 = concatenate([conv1, up8], axis=-1)

    conv8 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge8)
    conv8 = slice_at_block(conv8, 128 // 2, _slice_count, name='DABC_3')

    conv8 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv8 = TimeDistributed(Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv9 = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(conv8)

    model = Model(inputs=inputs, outputs=conv9)

    if load_weighted:
        model.load_weights(load_weighted)

    model.compile(optimizer=opt, loss=[weighted_dice_with_CE], metrics=['accuracy', dice_coef])

    return model


if __name__ == '__main__':
    model = DABC(input_size=(4, 256, 256, 1))
    model.summary()
