import keras.backend as K
from keras.losses import binary_crossentropy


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
