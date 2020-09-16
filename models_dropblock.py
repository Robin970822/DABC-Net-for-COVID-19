from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K

from keras.losses import binary_crossentropy
# from keras.losses import a
import tensorflow as tf

smooth = 0.001

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(
            keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(
            scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        # _, self.h, self.w, self.channel = input_shape.as_list()  # 元组，需要通过as_list()的操作转换成list.(tf中？)
        # https://blog.csdn.net/m0_37393514/article/details/82226754
        _, self.h, self.w, self.channel = input_shape[-4], input_shape[-3], input_shape[-2], input_shape[-1]
        # pad the mask
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
                             true_fn=lambda: output *
                             tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
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
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [
                              1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(
            keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(
            scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
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
                             true_fn=lambda: output *
                             tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
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
        mask = tf.nn.max_pool3d(mask, [
                                1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


def dice_coef(y_true, y_pred, smooth=1):  # 3D
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
    # return -dice_coef(y_true, y_pred)+CE
    # 02/13 修改  02/20 DICE:5  04/06 dice:1
    return 0.2 * (1 - dice_coef(y_true, y_pred)) + CE


def IoU(y_true, y_pred, eps=1e-6):
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + \
        K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + eps) / (union + eps), axis=0)


def iou_loss(in_gt, in_pred):
    return - IoU(in_gt, in_pred)


def resconv(inputlayer, outdim, name, is_batchnorm=True):
    # 只跨越了1层。两个卷积层+ relu + BN （Maxpooling在外部有）
    kinit = 'he_normal'
    x = TimeDistributed(Conv2D(outdim, 3, activation='relu', padding='same',
                               kernel_initializer=kinit, name=name + '_1'))(inputlayer)
    # if is_batchnorm:
    #     x = BatchNormalization(name=name + '_1_bn')(x)
    # x = Activation('relu', name=name + '_1_act')(x)

    x1 = TimeDistributed(Conv2D(outdim, 3, activation='relu',
                                padding='same', kernel_initializer=kinit, name=name + '_2'))(x)
    # x = Conv2D(outdim, (3, 3), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    x2 = Add()([x, x1])  # 只跨越了1层
    # x2 = Add()([inputlayer, x1])  # 跨越了2层，但此网络中不适用

    if is_batchnorm:
        x2 = TimeDistributed(BatchNormalization(
            axis=3), name=name + '_2_bn')(x2)
    # x = Activation('relu', name=name + '_2_act')(x)
    return x2


def slice_at_block(inputlayer, outdim, name='None'):
    # 接收两个concat后的特征图 : None,4,40,40,(256+256)
    # 输出后接卷积层
    x_0 = TimeDistributed(DepthwiseConv2D(3, 1, padding='same', activation='relu', name=name + '_dw'),
                          name='T_' + name + '_dw')(
        inputlayer)  # 缓冲层，每个slice每channel单独处理，输入输出不变
    # got None,4,40,40,512

    '''
    第三条支路，特征通道attention
    '''
    x_3 = TimeDistributed(GlobalAveragePooling2D(
    ), name='T_' + name + '_gap')(inputlayer)  # 由(4, 40, 40, 512) 得到 (4, 512)

    # spatial attention
    x = TimeDistributed(Conv2D(2, 1, padding='same', activation='relu', name=name + '_1*1'), name=name)(
        x_0)  # 04/03 原先的两条分支之一
    # 激活函数是否用/用sigmoid还是relu存疑！
    # got None,4,40,40,1
    x = TimeDistributed(BatchNormalization(
        axis=3, name=name + '_bn'), name='T_' + name + '_bn')(x)  # axis = 3 等价于 -1
    convfilter = 2
    x = ConvLSTM2D(filters=convfilter, kernel_size=(3, 3), padding='same', return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', activation='sigmoid', name=name + '_SABC')(x)
    # 最后的sigmoid激活函数包含在上面
    # 合并attention分支到主路特征图
    # x = K.repeat_elements(x, outdim, axis=-1)  #  报错！模型需要层封装，而不是数值。
    # x = Lambda(lambda x: K.repeat_elements(x, outdim // convfilter, axis=-1))(x)  # 将任意表达式封装为 Layer 对象。 04/22取消
    # 02/23 这里2 channel变 outdim（如512维）时，复制是按121212 而非111222.
    # 使用 矩阵传播 还是 1*1卷积 来reshape到原来形状，存疑！
    x = TimeDistributed(Conv2D(
        1, 3, padding='same', activation='sigmoid', name=name + '_3*3sig'))(x)  # 04/22
    # x = Multiply()([x, inputlayer])  # 对应元素乘。got None,4,40,40,512
    # 需要再加inputlayer吗？

    # channel attention add
    # x_3 = Lambda(lambda x: K.expand_dims(x, axis=2))(x_3)  # 04/08
    # x_3 = Lambda(lambda x: K.expand_dims(x, axis=2))(x_3)  # 04/08
    # x_3 = Lambda(lambda x: K.reshape(x, (4, 16, outdim//8)))(x_3)  # 04/22 4,1,1,512 -> 4,16,32,1 bug
    x_3 = Reshape((4, 16, outdim//16, 1))(x_3)  # 04/22 keras 有 reshape layers
    # x_3 = Lambda(lambda x: tf.transpose(x, perm=(0, 1, 2, 4, 3)))(x_3)  # 04/08 将特征通道转到图像尺寸
    # x_3 = Lambda(lambda x: tf.transpose(x, perm=(0,4,2,3,1)))(x_3)  # 04/07 1 cannot use K.transpose
    x_3 = ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=True, go_backwards=False,
                     kernel_initializer='he_normal', activation='sigmoid', name=name + '_CABC')(x_3)

    # x_3 = Lambda(lambda x: K.reshape(x, (4, 1, 1, outdim)))(x_3)  # 04/22  16,32,1 -> 1,1,512 bug
    x_3 = Reshape((4, 1, 1, outdim))(x_3)  # 04/22
    # x_3 = Lambda(lambda x: tf.transpose(x, perm=(0,4,2,3,1)))(x_3)  # 04/07 1
    # x_3 = Lambda(lambda x: tf.transpose(x, perm=(0, 1, 2, 4, 3)))(x_3)   # 04/08 还原 特征通道转到图像尺寸
    # x_3 = Multiply()([inputlayer, x_3])

    # 04/08
    x = Add()([x, x_3])
    x = TimeDistributed(Activation('sigmoid'))(x)

    x = Add()([x, inputlayer])
    return x


# 原(256,256,1) / ( batchs ,512,512,1)
def BCDU_net_D3(input_size=(10, 256, 256, 1), opt=Adam(lr=1e-4), load_weighted=None, is_trainable=True):
    slices = input_size[0]  # 10
    N = input_size[1]  # 原:0  N代表图的长宽：256（这里长和宽相等）
    droprate = 0.5  # 0.5

    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same',
                                   kernel_initializer='he_normal', input_shape=(slices, None, None, 1)))(inputs)
    conv1 = TimeDistributed(Conv2D(
        64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)
    # just contrast 3D and TimeDistributed  层的输入输出形状一致，只是在第三个维度上是否独立
    # conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    # conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    # conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    conv2 = resconv(pool1, 128, name='res_block1',
                    is_batchnorm=True)  # outdim 即 outfilter

    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    # conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    # conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
    conv3 = resconv(pool2, 256, name='res_block2', is_batchnorm=True)

    # drop3 = TimeDistributed(Dropout(droprate))(conv3, training=is_trainable)
    drop3 = TimeDistributed(DropBlock2D(0.5, 3))(
        conv3, training=is_trainable)  # 05/26

    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    # D1
    conv4 = TimeDistributed(Conv2D(512, 3, activation='relu',
                                   padding='same', kernel_initializer='he_normal'))(pool3)
    conv4_1 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
    # drop4_1 = TimeDistributed(Dropout(droprate))(conv4_1, training=is_trainable)
    drop4_1 = TimeDistributed(DropBlock2D(0.5, 3))(
        conv4_1, training=is_trainable)
    # D2
    conv4_2 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(drop4_1)
    conv4_2 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4_2)
    # conv4_2 = TimeDistributed(Dropout(droprate))(conv4_2, training=is_trainable)
    conv4_2 = TimeDistributed(DropBlock2D(0.5, 3))(
        conv4_2, training=is_trainable)
    # D3
    merge_dense = concatenate([conv4_2, drop4_1], axis=-1)
    # concatenate 层不需要加TimeDistributed。否则报错。注意叠加的通道要明确。这里-1=4 即第5个(None,10,64,64,512)->(...,1024)
    conv4_3 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge_dense)
    conv4_3 = TimeDistributed(Conv2D(
        512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4_3)
    # drop4_3 = TimeDistributed(Dropout(droprate))(conv4_3, training=is_trainable)
    drop4_3 = TimeDistributed(DropBlock2D(0.5, 3))(
        conv4_3, training=is_trainable)

    up6 = TimeDistributed(Conv2DTranspose(
        256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(drop4_3)
    up6 = TimeDistributed(BatchNormalization(axis=3))(up6)
    up6 = TimeDistributed(Activation('relu'))(up6)

    # No need to reshape
    # x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    # x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    # merge6  = concatenate([x1,x2], axis = 1)
    # merge6  = concatenate([drop3,up6], axis = 1)
    # 这里融合/处理有多种思路。
    # TODO  1按原文处理，lstm只整合skip connect和反卷积的特征图，即两个时间点  2(now!)lstm考虑切片维度的序列信息，即slices个时间点

    # need to reshape !  convLSTM expected ndim=5! 这里只能把(None，10，128，128，256)->(None，10，128*128，256)->(None，1,10，128*128，256)->concatenate
    # x1 = Reshape(target_shape=(1, slices, np.int32(N/4)*np.int32(N/4), 256))(drop3)
    # x2 = Reshape(target_shape=(1, slices, np.int32(N/4)*np.int32(N/4), 256))(up6)
    merge6 = concatenate([drop3, up6], axis=-1)  # 法二 此处只在特征通道上加,故1改为-1

    # 需要reshape回去. 注意形状！方法二中需要返回True序列。  # ConvLSTM2D(filters = 128
    # merge6 = ConvLSTM2D(filters = 512, kernel_size=(3, 3), padding='same', return_sequences = True, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
    # merge6 = Reshape(target_shape=(slices, np.int32(N/4), np.int32(N/4), 128))(merge6)  # to re reshape
    # merge6 = slice_at_block(merge6, 512, name='DABC_1')

    # conv6 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge6)
    # conv6 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv6)
    conv6 = resconv(merge6, 256, name='resblock3')
    conv6 = slice_at_block(conv6, 512//2, name='DABC_1')  # 04/22

    up7 = TimeDistributed(Conv2DTranspose(
        128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv6)
    up7 = TimeDistributed(BatchNormalization(axis=3))(
        up7)  # 不确定要不要加TimeDistributed
    up7 = TimeDistributed(Activation('relu'))(up7)

    # x1 = Reshape(target_shape=(1, slices, np.int32(N/2)*np.int32(N/2), 128))(conv2)
    # x2 = Reshape(target_shape=(1, slices, np.int32(N/2)*np.int32(N/2), 128))(up7)
    merge7 = concatenate([conv2, up7], axis=-1)  # (filters = 64,
    # merge7 = ConvLSTM2D(filters = 256, kernel_size=(3, 3), padding='same', return_sequences = True, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
    # merge7 = Reshape(target_shape=(slices, np.int32(N/2),np.int32(N/2), 64))(merge7)  # to re reshape
    # merge7 = slice_at_block(merge7, 256, name='DABC_2')

    # conv7 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge7)
    # conv7 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv7)
    conv7 = resconv(merge7, 128, name='resblock4')
    conv7 = slice_at_block(conv7, 256//2, name='DABC_2')  # 04/22

    up8 = TimeDistributed(Conv2DTranspose(
        64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'))(conv7)
    up8 = TimeDistributed(BatchNormalization(axis=3))(up8)
    up8 = TimeDistributed(Activation('relu'))(up8)

    # x1 = Reshape(target_shape=(1, slices, N* N, 64))(conv1)
    # x2 = Reshape(target_shape=(1, slices, N* N, 64))(up8)
    merge8 = concatenate([conv1, up8], axis=-1)  # (filters = 32,
    # merge8 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = True, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
    # merge8 = Reshape(target_shape=(slices, N, N, 32))(merge8)  # to re reshape
    # merge8 = slice_at_block(merge8, 128,name='DABC_3')

    conv8 = TimeDistributed(Conv2D(64, 3, activation='relu',
                                   padding='same', kernel_initializer='he_normal'))(merge8)
    conv8 = slice_at_block(conv8, 128//2, name='DABC_3')

    conv8 = TimeDistributed(Conv2D(
        64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv8 = TimeDistributed(Conv2D(
        2, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)
    conv9 = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(conv8)
    # conv9 = Dropout(0.5)(conv9, training=True)
    # conv9 = TimeDistributed(DropBlock2D(0.5, 3))(conv9, training=True)  # 05/26

    # model = Model(input = inputs, output = merge6)
    model = Model(input=inputs, output=conv9)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=opt, loss=[weighted_dice_with_CE], metrics=[
                  'accuracy', dice_coef])

    if load_weighted:
        model.load_weights(load_weighted)
    return model


# for plot

if __name__ == '__main__':
    # from keras.utils import plot_model
    a = BCDU_net_D3(input_size=(4, 256, 256, 1))
    a.summary()
    # plot_model(a, 'model3_V86.png', show_shapes=True)
