"""Created on Sun Aug 29 21:45:35 2021.

@author: malkhatib

"""

# external
from tensorflow.keras.layers import Conv2D, Conv3D, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def vdsr_2D(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c2 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    c4 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    c5 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    c6 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)
    c8 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)
    c9 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)
    c10 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)
    c11 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c10)
    c12 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c11)
    c13 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c12)
    c14 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c13)
    c15 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c14)
    c16 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c15)
    c17 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c16)
    c18 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c17)
    c19 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c18)
    c20 = Conv2D(
        IMG_CHANNELS,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c19)

    outputs = add([c20, s])

    model = Model(inputs=[s], outputs=[outputs])

    return model


def vdsr_3D(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))

    c1 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(s)
    c2 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c1)
    c3 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c2)
    c4 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c3)
    c5 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c4)
    c6 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c5)
    c7 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c6)
    c8 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c7)
    # c10 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    # c11 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    # c12 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
    # c13 = add([c12,c7])
    # c14 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
    # c15 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c14)
    # c16 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c15)
    # c17 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c16)
    # c18 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c17)
    # c19 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c18)
    # c20 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c19)
    # c21 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c20)
    # c22 = add([c21,c14])
    c23 = Conv3D(
        1,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c8)

    outputs = add([c23, s])

    model = Model(inputs=[s], outputs=[outputs])

    return model
