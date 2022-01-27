from dknn import *
from utils.utils_models import *
from utils.utils_data.get_data import get_data

from matplotlib import pyplot as plt
import os
import matplotlib
import numpy as np
import tensorflow as tf
import itertools
import pickle

import time
from utils.utils_model import make_cifar10_cnn
from utils.utils_data.get_data import get_data
from utils.utils_models.cifar10_models import MODELS


start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

# parameters for testing
amount_points = 30000
amount_calibration = 10
backend = NearestNeighbor.BACKEND.FAISS
#path_model = "/home/inafen/jupyter_notebooks/model_lnet5_2" #TODO
path_model = "/home/inafen/jupyter_notebooks/1004_model_cifar10_API_temp"
k_neighbors = 10

(X_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

train_data_DkNN = X_train[:amount_points]
train_labels_DkNN = y_train[:amount_points]
calibration_data = X_train[amount_points : (amount_points + 10) ] #TODO calibration data
calibration_labels = y_train[amount_points : (amount_points + 10) ]
fprop_data_DkNN = X_train[(amount_points + 10):]
fprop_labels_DkNN = y_train[(amount_points + 10):]

#temp----------------------------------------------------------------
#model = tf.keras.models.load_model(path_model)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

input_shape = (32, 32, 3)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test=x_test / 255.0

try:
    model = tf.keras.models.load_model(path_model)
except:
    # make lenet5 model for mnist dataset
    model = make_cifar10_cnn()
    # compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics="accuracy",
    )
    # train the model
    # if you want specify batch size, learning rates etc.
    history = model.fit(x_train, y_train, epochs=1, batch_size=128)
    # export model
    model.save(path_model)

# summarize filter shapes per layer
# we are only interested in convolutional layers
# print("Layer names (and shapes) of the model:")
layer_indices = []
nb_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if ("conv" not in layer.name) and ("dense" not in layer.name):
        # print(layer.name)
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    # print(layer.name, filters.shape)
    nb_layers.append(layer.name)
    layer_indices.append(i)
print(nb_layers)

# Define callable that returns a dictionary of all activations for a dataset
def get_activations(data) -> dict:
    """
    A callable that takes a np array and a layer name and returns its activations on the data.

    :param data: dataset
    :return: data_activations (dictionary of all activations for given dataset)
    """

    data_activations = {}
    # obtain all the predictions on the data by making a multi-output model
    outputs = [model.get_layer(name=layer).output for layer in nb_layers]
    model_mult = Model(inputs=model.inputs, outputs=outputs)

    # add dimension if only one data point
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=0)
        predictions = model_mult.predict(data)
    else:
        # use the model for predictions (returns a list)
        predictions = model_mult.predict(data)

    for i in range(len(predictions)):
        pred = predictions[i]
        layer = nb_layers[i]

        # number of samples
        num_samples = pred.shape[0]

        # given the first dimension, numpy reshape has to deduce the other shape
        reshaped_pred = pred.reshape(num_samples, -1)

        data_activations[layer] = reshaped_pred
    return data_activations
# Wrap the model into a DkNNModel
dknn = DkNNModel(
    neighbors=k_neighbors,
    layers=nb_layers,
    get_activations=get_activations,
    train_data= x_train,
    train_labels=y_train,
    back_end=backend,
)

# calibrate model
dknn.calibrate(
    calibration_data,
    calibration_labels,
)


