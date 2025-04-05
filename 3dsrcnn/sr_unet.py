# external
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Input,
    MaxPooling2D,
    MaxPooling3D,
    add,
    concatenate,
)
from tensorflow.keras.models import Model


def SR_3D_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    n = 16
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))
    s = inputs

    # Contraction path
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling3D((2, 2, 1))(c1)

    c2 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p1)
    c2 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c2 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c2 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    c2 = add([p1, c2])  # added
    p2 = MaxPooling3D((2, 2, 1))(c2)

    c3 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p2)
    c3 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c3)
    c3 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c3)
    c3 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)

    c3 = add([p2, c3])  # added
    p3 = MaxPooling3D((2, 2, 1))(c3)

    c4 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p3)
    c4 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c4)
    c4 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c4)
    c4 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    c4 = add([p3, c4])  # added
    p4 = MaxPooling3D(pool_size=(2, 2, 1))(c4)

    c5 = Conv3D(
        16 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p4)
    c5 = Conv3D(
        16 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)
    c5 = Conv3D(
        16 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)
    c5 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    c5 = add([p4, c5])

    # Expansive path
    u6 = Conv3DTranspose(8 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    c6 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)
    c6 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)

    u7 = Conv3DTranspose(4 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u7)
    c7 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c7)
    c7 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c7)

    u8 = Conv3DTranspose(2 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u8)
    c8 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c8)
    c8 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c8)

    u9 = Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)
    c9 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv3D(1, (1, 1, 1), activation="relu")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def SR_2D_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_FILTERS=16):
    n = NUM_FILTERS
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c1 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    c1 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    c2 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    c2 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    c2 = add([p1, c2])  # added
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    c3 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    c3 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)

    c3 = add([p2, c3])  # added
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    c4 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    c4 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    c4 = add([p3, c4])  # added
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        16 * n,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p4)
    c5 = Conv2D(
        16 * n,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)
    c5 = Conv2D(
        16 * n,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)
    c5 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    c5 = add([p4, c5])

    # Expansive path
    u6 = Conv2DTranspose(8 * n, (3, 3), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)
    c6 = Conv2D(
        8 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(4 * n, (3, 3), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)
    c7 = Conv2D(
        4 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(2 * n, (3, 3), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)
    c8 = Conv2D(
        2 * n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(n, (3, 3), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)
    c9 = Conv2D(
        n, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation="relu")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def SR_3D_unet_model_v2(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_FILTERS=16):
    n = NUM_FILTERS
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1))
    s = inputs

    # Contraction path
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling3D((2, 2, 1))(c1)

    c2 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p1)
    c2 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    c2 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)

    c2 = add([p1, c2])
    p2 = MaxPooling3D((2, 2, 1))(c2)

    c3 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p2)
    c3 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)

    c3 = add([p2, c3])
    p3 = MaxPooling3D((2, 2, 1))(c3)

    c4 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p3)
    c4 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    c4 = add([p3, c4])
    p4 = MaxPooling3D(pool_size=(2, 2, 1))(c4)

    c5 = Conv3D(
        16 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p4)
    c5 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    c5 = add([p4, c5])

    # Expansive path
    u6 = Conv3DTranspose(8 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    c6 = Conv3D(
        8 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)

    u7 = Conv3DTranspose(4 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u7)
    c7 = Conv3D(
        4 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c7)

    u8 = Conv3DTranspose(2 * n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u8)
    c8 = Conv3D(
        2 * n,
        (3, 3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c8)

    u9 = Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 1), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Conv3D(
        n, (3, 3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv3D(1, (1, 1, 1), activation="relu")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
