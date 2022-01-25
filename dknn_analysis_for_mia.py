from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils.utils_data import *
from utils.utils_model import *
from utils.utils_plot import *
from utils.utils import *


import os
import matplotlib
import numpy as np
import tensorflow as tf

import time

start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

# load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print("Shape of training data: {}".format(X_train.shape))
print("Shape of training labels: {}".format(y_train.shape))

# make lenet5 model for mnist dataset
model = make_lenet5_mnist_model()

# compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics="accuracy",
)

# train the model
# if you want specify batch size, learning rates etc.
r = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=1,
)

model.summary()

# summarize filter shapes per layer
# we are only interested in convolutional layers
print("Layer names (and shapes) of the model:")
layer_indices = []
nb_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]

    # check for convolutional layer
    if ("conv" not in layer.name) and ("dense" not in layer.name):
        print(layer.name)
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    nb_layers.append(layer.name)
    layer_indices.append(i)

# create the different datasets
train_data = X_train[:10000]
train_labels = y_train[:10000]

# Use a holdout of the test set to simulate calibration data for the DkNN.
# Number of calibration points for the DkNN
nb_cali = 10

cali_data = X_test[:nb_cali]
y_cali = y_test[:nb_cali]
cali_labels = y_cali

test_data = X_test[nb_cali:]
y_test = y_test[nb_cali:]

# noisy data (brightness)
noisy_data = brighten_images(X_test[nb_cali:])
y_noisy = y_test[nb_cali:]
noisy_labels = y_noisy

# Define callable that returns a dictionary of all activations for a dataset
def get_activations(data):
    """
    A callable that takes a np array and a layer name and returns its activations on the data.

    :param data: dataset
    :return: data_activations (dictionary of all activations for given dataset)
    """
    data_activations = {}

    # obtain all the predictions on the data by making a multi-output model
    outputs = [model.get_layer(name=layer).output for layer in nb_layers]
    model_mult = Model(inputs=model.inputs, outputs=outputs)

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
neighbors = 5

dknn = DkNNModel(
    neighbors=neighbors,
    layers=nb_layers,
    get_activations=get_activations,
    train_data=train_data,
    train_labels=train_labels,
    back_end=NearestNeighbor.BACKEND.FAISS,
)

print("Start model calibration")
dknn.calibrate(cali_data, cali_labels)
print("Calibrated the model")

# forward propagation through DkNN
amount_data = 10000
# for test data
_, knns_ind_test, knns_labels_test, knns_distances_test = dknn.fprop_np(
    test_data[:amount_data], get_knns=True
)
knns_indices_list_test = list(knns_ind_test.items())
knns_labels_list_test = list(knns_labels_test.items())
knns_distances_list_test = list(knns_distances_test.items())
# for train data
dknn_preds, knns_ind_train, knns_labels_train, knns_distances_train = dknn.fprop_np(
    train_data[:amount_data], get_knns=True
)
knns_indices_list_train = list(knns_ind_train.items())
knns_labels_list_train = list(knns_labels_train.items())
knns_distances_list_train = list(knns_distances_train.items())
# for noisy data
dknn_preds, knns_ind_noisy, knns_labels_noisy, knns_distances_noisy = dknn.fprop_np(
    noisy_data[:amount_data], get_knns=True
)
knns_indices_list_noisy = list(knns_ind_noisy.items())
knns_labels_list_noisy = list(knns_labels_noisy.items())
knns_distances_list_noisy = list(knns_distances_noisy.items())

# analyse changes knns
# get how many changes in knns have happened between two layers
_, _, differences_knns_total_training = get_differences_knns_btw_layers(
    len(train_data[:amount_data]), knns_indices_list_train
)
_, _, differences_knns_total_test = get_differences_knns_btw_layers(
    len(test_data[:amount_data]), knns_indices_list_test
)
_, _, differences_knns_total_noisy = get_differences_knns_btw_layers(
    len(noisy_data[:amount_data]), knns_indices_list_noisy
)
# get mean of knns changes btw. two layers
mean_knns_layers_train = get_mean_knns_layer(
    knns_indices_list_train, differences_knns_total_training
)
mean_knns_layers_test = get_mean_knns_layer(
    knns_indices_list_test, differences_knns_total_test
)
mean_knns_layers_noisy = get_mean_knns_layer(
    knns_indices_list_noisy, differences_knns_total_noisy
)
# plot changes
plot_changes_knns_3(
    mean_knns_layers_train,
    mean_knns_layers_test,
    mean_knns_layers_noisy,
    knns_indices_list_test,
    "/home/inafen/jupyter_notebooks/bar_changes_knns.png",
    "Layers (e.g. layer 1 stands for layer 0 --> layer 1)",
    "Mean of changes in neighbors between two layers",
    "Changes in knns btw. layers",
)

# analyse changes labels
# get how many changes in labels of knns have happened between two layers
_, _, differences_labels_total_training = get_differences_knns_btw_layers(
    len(train_data[:amount_data]), knns_labels_list_train
)
_, _, differences_labels_total_test = get_differences_knns_btw_layers(
    len(test_data[:amount_data]), knns_labels_list_test
)
_, _, differences_labels_total_noisy = get_differences_knns_btw_layers(
    len(noisy_data[:amount_data]), knns_labels_list_noisy
)
# get mean of label of knns changes btw. two layers
mean_labels_layers_train = get_mean_knns_layer(
    knns_indices_list_train, differences_labels_total_training
)
mean_labels_layers_test = get_mean_knns_layer(
    knns_indices_list_test, differences_labels_total_test
)
mean_labels_layers_noisy = get_mean_knns_layer(
    knns_indices_list_noisy, differences_labels_total_noisy
)
# plot changes
plot_changes_knns_3(
    mean_labels_layers_train,
    mean_labels_layers_test,
    mean_labels_layers_noisy,
    knns_labels_list_test,
    "/home/inafen/jupyter_notebooks/bar_changes_labels.png",
    "Layers (e.g. layer 1 stands for layer 0 --> layer 1)",
    "Mean of changes in labels of neighbors between two layers",
    "Changes in labels of knns btw. layers",
)

# analyse distances of knns
# get the distances to data point of knns, euclidean distance
distances_knns_all_test = get_distances_of_knns(
    len(test_data[:amount_data]), knns_distances_list_test
)
distances_knns_all_train = get_distances_of_knns(
    len(train_data[:amount_data]), knns_distances_list_train
)
distances_knns_all_noisy = get_distances_of_knns(
    len(noisy_data[:amount_data]), knns_distances_list_noisy
)
# get mean distance per layer
mean_distances_test = get_mean_distances_of_knns(
    distances_knns_all_test, knns_distances_list_test
)
mean_distances_train = get_mean_distances_of_knns(
    distances_knns_all_train, knns_distances_list_train
)
mean_distances_noisy = get_mean_distances_of_knns(
    distances_knns_all_noisy, knns_distances_list_noisy
)
# plot mean distances
plot_changes_knns_3(
    mean_distances_train,
    mean_distances_test,
    mean_distances_noisy,
    knns_distances_list_test,
    "/home/inafen/jupyter_notebooks/bar_distances.png",
    "Layers",
    "Mean of distances of neighbors for layer (euclidean distance)",
    "Mean distances of knns for layers",
)

print("--- %s seconds ---" % (time.time() - start_time))
