"""Created on Sun Jan  2 00:38:46 2022.

@author: malkhatib

"""

# external
from tensorflow.keras.layers import Conv3D, Input
from tensorflow.keras.models import Model


################################################################
def mei_3DCNN(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    s = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))

    l1 = Conv3D(
        64, (9, 9, 7), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)

    l2 = Conv3D(
        32, (1, 1, 1), activation="relu", kernel_initializer="he_normal", padding="same"
    )(l1)

    l3 = Conv3D(
        9, (1, 1, 1), activation="relu", kernel_initializer="he_normal", padding="same"
    )(l2)

    l4 = Conv3D(1, (5, 5, 3), kernel_initializer="he_normal", padding="same")(l3)

    outputs = l4

    model = Model(inputs=[s], outputs=[outputs])

    return model
