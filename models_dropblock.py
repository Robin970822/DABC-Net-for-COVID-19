from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *

from keras.losses import binary_crossentropy
# from keras.losses import a
import tensorflow as tf

smooth = 0.001


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    """
    [1] Ghiasi G, Lin T Y, Le Q V. Dropblock: A regularization method for convolutional networks[C]//Advances in Neural Information Processing Systems. 2018: 10727-10737.
    """
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape[-4], input_shape[-3], input_shape[-2], input_shape[-1]
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        p1 = (self.block_size - 1) // 2
        p0= (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock3D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        d, w, h = tf.to_float(self.d), tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = ((1. - self.keep_prob) * (d * w * h) / (self.block_size ** 3) /
                      ((d - self.block_size + 1) * (w - self.block_size + 1) * (h - self.block_size + 1)))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.d - self.block_size + 1,
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool3d(mask, [1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_with_CE(y_true, y_pred):
    CE = binary_crossentropy(y_true, y_pred)
    return -dice_coef(y_true, y_pred) + CE


def weighted_dice_with_CE(y_true, y_pred):
    CE = binary_crossentropy(y_true, y_pred)
    return 0.2 * (1 - dice_coef(y_true, y_pred)) + CE





def resconv(inputlayer, outdim, name, is_batchnorm=True):
    kinit = 'he_normal'
    x = TimeDistributed(Conv2D(outdim, 3, activation='relu', padding='same',
                               kernel_initializer=kinit, name=name + '_1'))(inputlayer)

    x1 = TimeDistributed(Conv2D(outdim, 3, activation='relu',
                                padding='same', kernel_initializer=kinit, name=name + '_2'))(x)
    x2 = Add()([x, x1])

    if is_batchnorm:
        x2 = TimeDistributed(BatchNormalization(
            axis=3), name=name + '_2_bn')(x2)
    return x2


def slice_at_block(inputlayer, outdim, name='None'):
    x_0 = TimeDistributed(DepthwiseConv2D(3, 1, padding='same', activation='relu', name=name + '_dw'),
                          name='T_' + name + '_dw')(
        inputlayer)

    x_3 = TimeDistributed(GlobalAveragePooling2D(
    ), name='T_' + name + '_gap')(inputlayer)

    x = TimeDistributed(Conv2D(2, 1, padding='same', activation='relu', name=name + '_1*1'), name=name)(
        x_0)
    x = TimeDistributed(BatchNormalization(
        axis=3, name=name + '_bn'), name='T_' + name + '_bn')(x)
    convfilter = 2
    x = ConvLSTM2D(filters=convfilter, kernel_size=(3, 3), padding='same', return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', activation='sigmoid', name=name + '_SABC')(x)
    x = TimeDistributed(Conv2D(
        1, 3, padding='same', activation='sigmoid', name=name + '_3*3sig'))(x)  # 04/22
    x_3 = Reshape((4, 16, outdim//16, 1))(x_3)  # 04/22 keras 有 reshape layers
    x_3 = ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=True, go_backwards=False,
                     kernel_initializer='he_normal', activation='sigmoid', name=name + '_CABC')(x_3)

    x_3 = Reshape((4, 1, 1, outdim))(x_3)

    x = Add()([x, x_3])
    x = TimeDistributed(Activation('sigmoid'))(x)

    x = Add()([x, inputlayer])
    return x


def DABC(input_size=(10, 256, 256, 1), opt=Adam(lr=1e-4), load_weighted=None, is_trainable=True):
    slices = input_size[0]
    droprate = 0.5

    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same',
                                   kernel_initializer='he_normal', input_shape=(slices, None, None, 1)))(inputs)
    conv1 = TimeDistributed(Conv2D(
        64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)

    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = resconv(pool1, 128, name='res_block1',
                    is_batchnorm=True)

    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = resconv(pool2, 256, name='res_block2', is_batchnorm=True)

    drop3 = TimeDistributed(DropBlock2D(0.5, 3))(conv3, training=is_trainable)  # 05/26

    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = TimeDistributed(Conv2D(512, 3, activation='relu',
                                   padding='same', kernel_initializer='he_normal'))(pool3)
    conv4_1 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
    drop4_1 = TimeDistributed(DropBlock2D(0.5, 3))(conv4_1, training=is_trainable)
    conv4_2 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(drop4_1)
    conv4_2 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4_2)
    conv4_2 = TimeDistributed(DropBlock2D(0.5, 3))(conv4_2, training=is_trainable)
    merge_dense = concatenate([conv4_2, drop4_1], axis=-1)
    conv4_3 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge_dense)
    conv4_3 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4_3)
    drop4_3 = TimeDistributed(DropBlock2D(0.5, 3))(conv4_3, training=is_trainable)

    up6 = TimeDistributed(Conv2DTranspose(
        256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(drop4_3)
    up6 = TimeDistributed(BatchNormalization(axis=3))(up6)
    up6 = TimeDistributed(Activation('relu'))(up6)

    merge6 = concatenate([drop3, up6], axis=-1)


    conv6 = resconv(merge6, 256, name='resblock3')
    conv6 = slice_at_block(conv6, 512//2, name='DABC_1')

    up7 = TimeDistributed(Conv2DTranspose(
        128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv6)
    up7 = TimeDistributed(BatchNormalization(axis=3))(
        up7)
    up7 = TimeDistributed(Activation('relu'))(up7)

    merge7 = concatenate([conv2, up7], axis=-1)

    conv7 = resconv(merge7, 128, name='resblock4')
    conv7 = slice_at_block(conv7, 256//2, name='DABC_2')

    up8 = TimeDistributed(Conv2DTranspose(
        64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv7)
    up8 = TimeDistributed(BatchNormalization(axis=3))(up8)
    up8 = TimeDistributed(Activation('relu'))(up8)

    merge8 = concatenate([conv1, up8], axis=-1)

    conv8 = TimeDistributed(Conv2D(64, 3, activation='relu',
                                   padding='same', kernel_initializer='he_normal'))(merge8)
    conv8 = slice_at_block(conv8, 128//2, name='DABC_3')

    conv8 = TimeDistributed(Conv2D(
        64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv8 = TimeDistributed(Conv2D(
        2, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv9 = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    model.compile(optimizer=opt, loss=[weighted_dice_with_CE], metrics=[
                  'accuracy', dice_coef])

    if load_weighted:
        model.load_weights(load_weighted)
    return model


if __name__ == '__main__':
    model = DABC(input_size=(4, 256, 256, 1))
    model.summary()
