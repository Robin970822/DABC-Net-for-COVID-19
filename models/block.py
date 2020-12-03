import tensorflow as tf
from keras.layers import *
from keras import backend as K

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
        p0 = (self.block_size - 1) - p1
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
        mask = tf.nn.max_pool3d(mask, [1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1],
                                'SAME')
        mask = 1 - mask
        return mask


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


def slice_at_block(inputlayer, outdim, _slice_count=4, name='None'):
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
        1, 3, padding='same', activation='sigmoid', name=name + '_3*3sig'))(x)
    x_3 = Reshape((_slice_count, 16, outdim // 16, 1))(x_3)
    x_3 = ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=True, go_backwards=False,
                     kernel_initializer='he_normal', activation='sigmoid', name=name + '_CABC')(x_3)

    x_3 = Reshape((_slice_count, 1, 1, outdim))(x_3)

    x = Add()([x, x_3])
    x = TimeDistributed(Activation('sigmoid'))(x)

    x = Add()([x, inputlayer])
    return x

def slice_with_block(inputlayer, outdim, _slice_count=4, name='None'):
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
    x_1 = ConvLSTM2D(filters=convfilter, kernel_size=(3, 3), padding='same', return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', activation='sigmoid')(x)
    x_2 = ConvLSTM2D(filters=convfilter, kernel_size=(3, 3), padding='same', return_sequences=True, go_backwards=True,
                   kernel_initializer='he_normal', activation='sigmoid')(x)
    x = Add()([x_1, x_2])
    x = TimeDistributed(Conv2D(
        1, 3, padding='same', activation='sigmoid', name=name + '_3*3sig'))(x)
    x_3 = Reshape((_slice_count, 16, outdim // 16, 1))(x_3)
    x_3 = ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=True, go_backwards=False,
                     kernel_initializer='he_normal', activation='sigmoid', name=name + '_CABC')(x_3)

    x_3 = Reshape((_slice_count, 1, 1, outdim))(x_3)

    x = Add()([x, x_3])
    x = TimeDistributed(Activation('sigmoid'))(x)

    x = Add()([x, inputlayer])
    return x
