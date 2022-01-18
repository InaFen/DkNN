from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils.utils_model import *

import os
import matplotlib
import numpy as np
import tensorflow as tf
import itertools
import pickle

import time

start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

K_NEIGHBORS = 10 #[5, 10, 50, 100]  # for DkNN
AMOUNT_GENERATE_NEIGHBORS = 200 #[50, 100, 200]  # for generate_neighboring_points
SCALES_EPSILON = [(0.005,0.2), (0.01, 0.3), (0.02, 0.4), (0.05,0.7), (0.1,1.5), (0.2,3.0)]  #TODO validate for different scales? # for generate_neighboring_points
AMOUNT_DATA_TOTAL = 100 # how many elements are used

# parameters for testing
amount_points = 30000
amount_calibration = 10
backend = (
    NearestNeighbor.BACKEND.FAISS
)
path_model = "/home/inafen/jupyter_notebooks/model_lnet5_2"

# load and preprocess MNIST data
# get data only out of training set
mnist = tf.keras.datasets.mnist
(X_train, y_train), _ = mnist.load_data()
X_train = X_train / 255

X_train = np.expand_dims(X_train, axis=-1)

# create the member, non member datasets
member_data = X_train[:amount_points]
member_labels = y_train[:amount_points]
non_member_data = X_train[amount_points : (amount_points * 2)]
non_member_labels = y_train[amount_points : (amount_points * 2)]


print("Shape of member data: {}".format(member_data.shape))
print("Shape of member labels: {}".format(member_labels.shape))
print("Shape of non member data: {}".format(non_member_data.shape))
print("Shape of non member labels: {}".format(non_member_labels.shape))

#def build_model(member_data = member_data, member_labels = member_labels, non_member_data = non_member_data, non_member_labels = non_member_labels, path_model = path_model): #TODO does it work as function? Or in code itself?
#create and train model-----------------------------------------------------------------------
try:
    model = tf.keras.models.load_model(path_model)
except:
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
    history = model.fit(member_data, member_labels, epochs=12, batch_size=128)
    # export model
    model.save(path_model)

# model.summary()
train_accuracy = model.evaluate(member_data, member_labels)
test_accuracy = model.evaluate(non_member_data, non_member_labels)
# print("Train accuracy: {}".format(train_accuracy[1]))
# print("Test accuracy: {}".format(test_accuracy[1]))

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

# Define callable that returns a dictionary of all activations for a dataset --------------------
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

def experiments_setup_DkNN(train_data_DkNN, train_labels_DkNN, calibration_data, calibration_label, data_fprop_DkNN, labels_fprop_DkNN, filepath_pickle = "/home/inafen/jupyter_notebooks/validate_DkNN_0.pickle", get_activations=get_activations, k_neighbors=K_NEIGHBORS, backend= backend, nb_layers = nb_layers):

    all_data_one_experiment_for_pickle = {}

    # Wrap the model into a DkNNModel
    # DkNN for Member datapoint
    dknn = DkNNModel(
        neighbors=k_neighbors,
        layers=nb_layers,
        get_activations=get_activations,
        train_data=train_data_DkNN,
        train_labels=train_labels_DkNN,
        back_end=backend,
    )

    # calibrate model
    dknn.calibrate(
        calibration_data,
        calibration_label,
    )
    for element in range(AMOUNT_DATA_TOTAL):
        print("element number: {}".format(element))
        # forward propagation through DkNNs
        _, knns_ind, knns_labels, knns_distances = dknn.fprop_np(
            data_fprop_DkNN[element], get_knns=True
        )

        current_data_for_pickle = {}

        current_data_for_pickle["train_accuracy"] = train_accuracy
        current_data_for_pickle["test_accuracy"] = test_accuracy
        current_data_for_pickle["label of element"] = labels_fprop_DkNN[element]
        current_data_for_pickle["knns_ind"] = knns_ind
        current_data_for_pickle["knns_labels"] = knns_labels
        current_data_for_pickle["knns_distances"] = knns_distances
        current_data_for_pickle["k_neighbors"] = k_neighbors
        current_data_for_pickle["amount_generate_neighbors"] = AMOUNT_GENERATE_NEIGHBORS
        current_data_for_pickle["AMOUNT_M_NM_TOTAL"] = AMOUNT_DATA_TOTAL
        all_data_one_experiment_for_pickle["element {}".format(element)] = current_data_for_pickle

    with open(filepath_pickle, "wb") as f:
        pickle.dump(all_data_one_experiment_for_pickle, f)

#experiment 1.1
experiments_setup_DkNN(member_data, member_labels, non_member_data[:amount_calibration], non_member_labels[:amount_calibration], non_member_data[amount_calibration:amount_calibration+100],non_member_labels[amount_calibration:amount_calibration+100], "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_1.pickle")

with open("/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_1.pickle", "rb") as f:
    loaded_obj = pickle.load(f)
print(loaded_obj)

for element in range(100):
    print(loaded_obj["element {}".format(element)]["label of element"])
    print(loaded_obj["element {}".format(element)]["knns_labels"])

print("--- %s seconds ---" % (time.time() - start_time))
