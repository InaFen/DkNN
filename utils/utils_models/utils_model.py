from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Flatten,
    Dense,
    Dropout,
    MaxPooling2D,
    AveragePooling2D,
    UpSampling2D,
    Softmax,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
import tensorflow as tf


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


def make_cifar10_cnn(
    from_logits: bool = True, training: bool = False
) -> "keras.engine.functional.Functional":
    """
    CIFAR 10 CNN model, based on Model subclass (https://github.com/fraboeni/membership-risk/blob/master/code_base/models.py)

    :param from_logits: If False, last Dense layer has activation='softmax'
    :param training: If True, Dropout layers are added
    :return: model
    """

    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3))(
        i
    )
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    if training:
        x = Dropout(0.2)(x, training=training)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x, training=training)
    if training:
        x = Dropout(0.2)(x, training=training)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    if training:
        x = Dropout(0.2)(x, training=training)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    if training:
        x = Dropout(0.2)(x)
    if from_logits:
        x = Dense(10)(x)
    else:
        x = Dense(10, activation="softmax")(x)
    model = Model(i, x)
    return model


def make_cifar10_resnet50(
    base_trainable: bool = True, from_logits: bool = True
) -> "keras.engine.functional.Functional":
    """
     CIFAR 10 ResNet50 model, based on Model subclass (https://github.com/fraboeni/membership-risk/blob/master/code_base/models.py)

    :param base_trainable: If False, feature_extractor.trainable = False
    :param from_logits: If False, last Dense layer has activation='softmax'
    :return: model
    """
    feature_extractor = tf.keras.applications.resnet.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = UpSampling2D(size=(7, 7))(
        i
    )  # upsample 32, 32 to 224, 224 by multiplying with factor 7
    if not base_trainable:
        feature_extractor.trainable = False
    x = feature_extractor(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    if from_logits:
        x = Dense(10)(x)
    else:
        x = Dense(10, activation="softmax")(x)
    model = Model(i, x)
    return model
