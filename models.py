from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *        

from keras.losses import binary_crossentropy
# from keras.losses import a
import keras.backend as K

smooth = 0.001


def dice_coef(y_true, y_pred, smooth=1):  # 3D
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_loss_with_CE(y_true, y_pred):
    CE=binary_crossentropy(y_true,y_pred)
    return -dice_coef(y_true, y_pred)+CE

def weighted_dice_with_CE(y_true, y_pred):
    CE=binary_crossentropy(y_true,y_pred)
    return 0.2*(1-dice_coef(y_true, y_pred))+CE



def resconv(inputlayer, outdim, name, is_batchnorm=True):
    kinit = 'he_normal'
    x = TimeDistributed(Conv2D(outdim, 3, activation = 'relu', padding = 'same', kernel_initializer=kinit, name=name + '_1'))(inputlayer)


    x1 = TimeDistributed(Conv2D(outdim, 3, activation = 'relu', padding = 'same', kernel_initializer=kinit, name=name + '_2'))(x)
    x2 = Add()([x, x1])

    if is_batchnorm:
        x2 = TimeDistributed(BatchNormalization(axis=3), name=name + '_2_bn')(x2)
    return x2



def slice_at_block(inputlayer, outdim, name='None'):

    x_0 = TimeDistributed(DepthwiseConv2D(3, 1, padding='same', activation='relu', name=name + '_dw'),
                          name='T_' + name + '_dw')(
        inputlayer)


    x_3 = TimeDistributed(GlobalAveragePooling2D(), name='T_' + name + '_gap')(inputlayer)

    x = TimeDistributed(Conv2D(2, 1, padding='same', activation='relu', name=name + '_1*1'), name=name)(
        x_0)
    x = TimeDistributed(BatchNormalization(axis=3, name=name + '_bn'), name='T_' + name + '_bn')(x)
    convfilter = 2
    x = ConvLSTM2D(filters=convfilter, kernel_size=(3, 3), padding='same', return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', activation='sigmoid', name=name + '_SABC')(x)
    x = TimeDistributed(Conv2D(1, 3, padding='same', activation='sigmoid', name=name + '_3*3sig'))(x)


    x_3 = Reshape((4,16, outdim//16,1))(x_3)
    x_3 = ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=True, go_backwards=False,
                     kernel_initializer='he_normal', activation='sigmoid', name=name + '_CABC')(x_3)

    x_3 = Reshape((4, 1, 1, outdim))(x_3)

    x = Add()([x, x_3])
    x = TimeDistributed(Activation('sigmoid'))(x)

    x = Add()([x, inputlayer])
    return x

def DABC(input_size = (10,256,256,1), opt = Adam(lr = 1e-4),load_weighted = None,):
    slices =  input_size[0]
    droprate = 0.5
    is_training = False

    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',input_shape=(slices,None,None,1)))(inputs)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)


    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = resconv(pool1, 128, name='res_block1',is_batchnorm=True)

    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = resconv(pool2, 256, name='res_block2', is_batchnorm=True)

    drop3 = TimeDistributed(Dropout(droprate))(conv3, training=False)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv4_1 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)
    drop4_1 = TimeDistributed(Dropout(droprate))(conv4_1, training=False)
    conv4_2 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(drop4_1)
    conv4_2 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4_2)
    conv4_2 = TimeDistributed(Dropout(droprate))(conv4_2, training=False)
    merge_dense = concatenate([conv4_2,drop4_1], axis = -1)
    conv4_3 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge_dense)
    conv4_3 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4_3)
    drop4_3 = TimeDistributed(Dropout(droprate))(conv4_3, training=False)

    up6 = TimeDistributed(Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal'))(drop4_3)
    up6 = TimeDistributed(BatchNormalization(axis=3))(up6)
    up6 = TimeDistributed(Activation('relu'))(up6)

    merge6 = concatenate([drop3, up6], axis = -1)

    conv6 = resconv(merge6, 256 ,name='resblock3')
    conv6 = slice_at_block(conv6, 512//2, name='DABC_1')

    up7 = TimeDistributed(Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal'))(conv6)
    up7 = TimeDistributed(BatchNormalization(axis=3))(up7)
    up7 = TimeDistributed(Activation('relu'))(up7)

    merge7 = concatenate([conv2,up7], axis = -1)
    conv7 = resconv(merge7, 128, name='resblock4')
    conv7 = slice_at_block(conv7, 256//2, name='DABC_2')

    up8 = TimeDistributed(Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal'))(conv7)
    up8 = TimeDistributed(BatchNormalization(axis=3))(up8)
    up8 = TimeDistributed(Activation('relu'))(up8)

    merge8 = concatenate([conv1,up8], axis = -1)

    conv8 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge8)
    conv8 = slice_at_block(conv8, 128//2,name='DABC_3')

    conv8 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv8)
    conv8 = TimeDistributed(Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv8)
    conv9 = TimeDistributed(Conv2D(1, 1, activation = 'sigmoid'))(conv8)

    model = Model(inputs = inputs, outputs = conv9)

    if load_weighted:
        model.load_weights(load_weighted)

    model.compile(optimizer = opt, loss =[weighted_dice_with_CE], metrics = ['accuracy',dice_coef])

    return model
    

if __name__ == '__main__':

    model=DABC(input_size = (4,256,256,1))
    model.summary()

