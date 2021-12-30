from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from dknn import *
from utils_model import *
from utils_plot import *

import os
import matplotlib
import numpy as np
import tensorflow as tf
from six.moves import xrange

import time

start_time = time.time()

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
# Use a holdout of the test set to simulate calibration data for the DkNN.
train_data = X_train[:10000]
train_labels = y_train[:10000]
print("Shape of training dataset: {}{}".format(train_data.shape, train_labels.shape))

# Number of calibration points for the DkNN
nb_cali = 10

cali_data = X_test[:nb_cali]
y_cali = y_test[:nb_cali]
cali_labels = y_cali
print("Shape of calibration dataset: {}{}".format(cali_data.shape, y_cali.shape))

test_data = X_test[nb_cali:]
y_test = y_test[nb_cali:]
print("Shape of test dataset: {}{}".format(test_data.shape, y_test.shape))

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
)

print("Start model calibration")
dknn.calibrate(cali_data, cali_labels)
print("Calibrated the model")
print("")
print("")
print("")


# Test the DkNN on clean test data and FGSM test data
# run DkNN first on test data
# then run DkNN on cali data

# Test the DkNN on clean test data
amount_data = 4
for data_in, fname in zip([test_data[:amount_data]], ["test"]):
    print("Shape of data: {} (from dataset {})".format(data_in.shape, fname))
    dknn_preds = dknn.fprop_np(data_in)
    print("Shape of credibility: {}".format(dknn_preds.shape))
    print(
        "Mean of predicted labels = true labels: {}".format(
            np.mean(np.argmax(dknn_preds, axis=1) == (y_test[:amount_data]))
        )
    )
    plot_reliability_diagram(
        dknn_preds, y_test[:amount_data], "/tmp/dknn_" + fname + ".pdf"
    )

for data_in, fname in zip([cali_data[:amount_data]], ["cali"]):
    print("Shape of data: {} (from dataset {})".format(data_in.shape, fname))
    dknn_preds = dknn.fprop_np(data_in)
    print("Shape of credibility: {}".format(dknn_preds.shape))
    print(
        "Mean of predicted labels = true labels: {}".format(
            np.mean(np.argmax(dknn_preds, axis=1) == (cali_labels[:amount_data]))
        )
    )
    plot_reliability_diagram(
        dknn_preds, cali_labels[:amount_data], "/tmp/dknn_" + fname + ".pdf"
    )

dknn.query_objects

print("--- %s seconds ---" % (time.time() - start_time))
