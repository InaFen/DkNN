from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Flatten,
    Dense,
    Dropout,
    MaxPooling2D,
    AveragePooling2D,
    Activation,
    Softmax,
)
from tensorflow.keras.models import Model


def make_basic_cnn():
    """Build a basic CNN.

    :return: CNN model
    """
    shape = (28, 28, 1)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(i)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(10)(x)
    # leave out the last Softmax in order to have logits and no softmax
    # x = Activation('softmax')(x)
    model = Model(i, x)
    return model


def make_shallow_basic_cnn():
    """Build a basic CNN which lacks complexity.
    The reason for this is to have a fast CNN for testing purposes.

    :return: CNN model
    """
    shape = (28, 28, 1)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(i)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    model = Model(i, x)
    return model


def make_lenet5_mnist_model(activation=False):
    """
    Lenet-5 Implementation for MNIST:
    https://colab.research.google.com/drive/1CVm50PGE4vhtB5I_a_yc4h5F-itKOVL9#scrollTo=zLdfGt_GlP0x

    :param activation: if True, add softmax layer
    :return: model
    """
    shape = (28, 28, 1)
    inputs = Input(shape=shape)
    out1 = Conv2D(
        filters=6, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
    )(inputs)
    out2 = AveragePooling2D()(out1)
    out3 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(out2)
    out4 = AveragePooling2D()(out3)
    out5 = Flatten()(out4)
    out6 = Dense(units=120, activation="relu")(out5)
    out7 = Dense(units=84, activation="relu")(out6)
    pred = Dense(units=10)(out7)
    if activation:
        pred = Softmax()(pred)
    model = Model(inputs=inputs, outputs=pred)
    return model
