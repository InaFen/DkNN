from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils.utils_models.utils_model import *
from utils.utils_neighbors import generate_neighboring_points

from matplotlib import pyplot as plt
import os
import matplotlib
import numpy as np
import tensorflow as tf
import pickle

import time

start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

k_neighbors = 200  # for DkNN
amount_fprop_dknn = 100  # how many elements are forwarded through DkNN

# parameters for testing
amount_points = 30000 #how many points the DkNN is trained on
amount_calibration = 10
backend = NearestNeighbor.BACKEND.FALCONN #TODO do with FALCONN?
path_model = "/home/inafen/jupyter_notebooks/model_lnet5_2"
path_pickle = "/home/inafen/jupyter_notebooks/labels_PATE_DkNN_3.pickle" #pickle where DkNN details and output from fprop is saved
#"/home/inafen/jupyter_notebooks/labels_PATE_DkNN_1.pickle" has k_neighbors = 100
#"/home/inafen/jupyter_notebooks/labels_PATE_DkNN_2.pickle" with FAISS

# load and preprocess MNIST data
# get data only out of training set
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print("Shape of training data: {}".format(X_train.shape))
print("Shape of training labels: {}".format(y_train.shape))

# create the member, non member datasets
member_data = X_train[:amount_points]
member_labels = y_train[:amount_points]
non_member_data = X_test[:amount_points]
non_member_labels = y_test[:amount_points]


# create and train model-----------------------------------------------------------------------
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
#train_accuracy = model.evaluate(member_data, member_labels) #TODO uncomment
#test_accuracy = model.evaluate(non_member_data, non_member_labels)
#print("Train accuracy: {}".format(train_accuracy[1]))
#print("Test accuracy: {}".format(test_accuracy[1]))

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


def create_fprop_DkNN(
    train_data_DkNN,
    train_labels_DkNN,
    calibration_data,
    calibration_label,
    data_fprop_DkNN,
    labels_fprop_DkNN,
    filepath_pickle="/home/inafen/jupyter_notebooks/temp.pickle",
    get_activations=get_activations,
    k_neighbors=k_neighbors,
    backend=backend,
    nb_layers=nb_layers,
    save_pickle=True,
    amount_data_total=amount_fprop_dknn,
):

    #all_data_one_experiment_for_pickle = {}
    all_data_for_pickle = {}

    # Wrap the model into a DkNNModel
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
    """
    for element in range(amount_data_total):
        # data_fprop_DkNN is not an array when only one element is passed
        if amount_data_total == 1:
            data_fprop_DkNN_element = data_fprop_DkNN
            labels_fprop_DkNN_element = labels_fprop_DkNN
        else:
            data_fprop_DkNN_element = data_fprop_DkNN[element]
            labels_fprop_DkNN_element = labels_fprop_DkNN[element]
        # print("element number: {}".format(element))
        # forward propagation through DkNNs
        _, knns_ind, knns_labels, knns_distances = dknn.fprop_np(
            data_fprop_DkNN_element, get_knns=True
        )

        current_data_for_pickle = {}

        current_data_for_pickle["train_accuracy"] = train_accuracy
        current_data_for_pickle["test_accuracy"] = test_accuracy
        current_data_for_pickle["element"] = data_fprop_DkNN_element
        current_data_for_pickle["label of element"] = labels_fprop_DkNN_element
        current_data_for_pickle["knns_ind"] = knns_ind
        current_data_for_pickle[
            "data including knns (= train data DkNN)"
        ] = train_data_DkNN
        current_data_for_pickle["knns_labels"] = knns_labels
        current_data_for_pickle["knns_distances"] = knns_distances
        current_data_for_pickle["k_neighbors"] = k_neighbors
        current_data_for_pickle["AMOUNT_M_NM_TOTAL"] = amount_data_total
        all_data_one_experiment_for_pickle[
            "element {}".format(element)
        ] = current_data_for_pickle
    """
    #send all data at once through DkNN, so in one batch
    _, knns_ind, knns_labels, knns_distances = dknn.fprop_np(
        data_fprop_DkNN, get_knns=True
    )
    #all_data_for_pickle["train_accuracy"] = train_accuracy #TODO uncomment
    #all_data_for_pickle["test_accuracy"] = test_accuracy
    all_data_for_pickle["element"] = data_fprop_DkNN
    all_data_for_pickle["label of elements"] = labels_fprop_DkNN
    all_data_for_pickle["knns_ind"] = knns_ind
    all_data_for_pickle[
        "data including knns (= train data DkNN)"
    ] = train_data_DkNN
    all_data_for_pickle["knns_labels"] = knns_labels
    all_data_for_pickle["knns_distances"] = knns_distances
    all_data_for_pickle["k_neighbors"] = k_neighbors
    all_data_for_pickle["amount_data_total"] = amount_data_total

    if save_pickle == True:
        with open(filepath_pickle, "wb") as f:
            pickle.dump(all_data_for_pickle, f)
    else:
        return all_data_for_pickle

    """
    if save_pickle == True:
        with open(filepath_pickle, "wb") as f:
            pickle.dump(all_data_one_experiment_for_pickle, f)
    else:
        return all_data_one_experiment_for_pickle
    """

if not(os.path.isfile(path_pickle)):
    # create DkNN and fprop amount_fprop_dknn elements at once
    create_fprop_DkNN(
        member_data,
        member_labels,
        non_member_data[:amount_calibration],
        non_member_labels[:amount_calibration],
        non_member_data[amount_calibration : amount_calibration + amount_fprop_dknn],
        non_member_labels[amount_calibration : amount_calibration + amount_fprop_dknn],
        path_pickle,
    )

with open(
        path_pickle, "rb"
) as f:
    loaded_obj = pickle.load(f)
print("Pickle is loaded.")

print((loaded_obj["knns_labels"]['conv2d'][0]))