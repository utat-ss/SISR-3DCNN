"""Created on Sun Aug 29 21:02:35 2021.

@author: malkhatib

"""

# external
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Conv3D,
    Dense,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    Input,
    Reshape,
    add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def srcnn_2D_915(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    l1 = Conv2D(
        64,
        (9, 9),
        activation="relu",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    l2 = Conv2D(
        32, (1, 1), activation="relu", kernel_initializer="he_normal", padding="same"
    )(l1)
    l3 = Conv2D(
        IMG_CHANNELS,
        (5, 5),
        activation="linear",
        kernel_initializer="he_normal",
        padding="same",
    )(l2)
    outputs = l3
    # print(outputs)
    # print(s)
    model = Model(inputs=[s], outputs=[outputs])

    return model


def srcnn_2D_333(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    l1 = Conv2D(
        64,
        (3, 3),
        activation="relu",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    l2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(l1)
    l3 = Conv2D(
        IMG_CHANNELS,
        (3, 3),
        activation="linear",
        kernel_initializer="he_normal",
        padding="same",
    )(l2)
    outputs = l3
    # print(outputs)
    # print(s)
    model = Model(inputs=[s], outputs=[outputs])

    return model


def srcnn_3D_333(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))
    c1 = Conv3D(
        64, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c2 = Conv3D(
        32, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c3 = Conv3D(
        1, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    outputs = c3
    model = Model(inputs=[s], outputs=[outputs])

    return model


def srcnn_3D_915(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))
    c1 = Conv3D(
        64, (9, 9, 9), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c2 = Conv3D(
        32, (1, 1, 1), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c3 = Conv3D(
        1, (5, 5, 5), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    outputs = c3

    model = Model(inputs=[s], outputs=[outputs])

    return model


def srcnn_3D_333_residual(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))
    c1 = Conv3D(
        1,
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
        1,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        activity_regularizer=l2(10e-10),
    )(c4)

    # outputs = c3
    outputs = add([c5, s])
    model = Model(inputs=[s], outputs=[outputs])

    return model


def hydra_sr(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # input1
    input1 = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))

    # Spectral Feature Extractor
    # c1=Conv3D(64,(1,1,3),activation="relu", padding = "same")(input1)
    # c1=Conv3D(32,(1,1,3),activation="relu", padding = "same")(c1)

    # Spatial Feature Extractor
    # c2=Conv3D(64,(3,3,1),activation="relu", padding = "same")(input1)
    # c2=Conv3D(64,(3,3,1),activation="relu", padding = "same")(c2)

    # Spectral-Spatail Feature Extractor
    # c3=Conv3D(64,(3,3,3),activation="relu", padding = "same")(input1)
    # c3=Conv3D(64,(3,3,3),activation="relu", padding = "same")(c3)

    # result = add([c1,c2,c3]);

    # Feature Fusion
    # result = tf.concat([c1,c2,c3], axis = 4);

    outputs = Conv3D(1, (3, 3, 3), activation="relu", padding="same")(input1)

    model = Model(inputs=[input1], outputs=[outputs])
    # model.compile(tf.keras.optimizers.Adam(1e-4),
    # loss='mse' ,
    # metrics= [ssim, psnr, tf.keras.metrics.CosineSimilarity(axis=-1)])
    return model


def se_srcnn_3D(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))

    c1 = Conv3D(
        64, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c2 = Conv3D(
        64, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c3 = Conv3D(
        64, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    se = SE_Block_3D(c3)

    outputs = Conv3D(
        1, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(se)

    model = Model(inputs=[s], outputs=[outputs])

    return model


def SE_Block_2D(xin):
    # Squeeze Path
    r = Reshape((xin.shape[1], xin.shape[2], xin.shape[3] * xin.shape[4]))(xin)

    GaP = GlobalAveragePooling2D()(r)
    sqz_layer = Dense(units=r.shape[-1] // xin.shape[3], activation="relu")(GaP)

    # Excitation Path
    excite_layer1 = Dense(units=r.shape[-1], activation="sigmoid")(sqz_layer)
    excite_layer2 = tf.keras.layers.multiply([r, excite_layer1])

    out = Reshape((xin.shape[1], xin.shape[2], xin.shape[3], xin.shape[4]))(
        excite_layer2
    )

    return out


def SE_Block_3D(xin):
    # Squeeze Path
    GaP = GlobalAveragePooling3D()(xin)
    sqz_layer = Dense(units=xin.shape[-1] // 16, activation="relu")(GaP)

    # Excitation Path
    excite_layer1 = Dense(units=xin.shape[-1], activation="sigmoid")(sqz_layer)
    # excite_layer1 = Reshape((xin.shape[1], xin.shape[2], xin.shape[3], xin.shape[4]))(excite_layer1)

    excite_layer2 = tf.keras.layers.multiply([xin, excite_layer1])

    out = excite_layer2

    return out
