from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils_model import *
from utils_neighbors import *


import os
import matplotlib
import numpy as np
import tensorflow as tf
import itertools
import pickle

import faiss

import time

start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

#parameters for testing
amount_points = 30000
amount_calibration = 1000 #TODO change?
backend = NearestNeighbor.BACKEND.FALCONN
#amount_data_fprop = 1000 #amount of data that should be used for forward propagation through DkNN

SCALES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
K_NEIGHBORS = [5, 10, 20, 50, 100, 200] #for DkNN
AMOUNT_GENERATE_NEIGHBORS = [10, 50, 100, 200, 500] #for generate_neighboring_points

AMOUNT_M_NM_TOTAL = 1000 #1000 #how many of member/non member elements are used

hyperparamters = []
hyperparamters.append(SCALES)
hyperparamters.append(K_NEIGHBORS)
hyperparamters.append(AMOUNT_GENERATE_NEIGHBORS)
HYPERPARAMETERS = list(itertools.product(*hyperparamters))

# load and preprocess MNIST data
#get data only out of training set
mnist = tf.keras.datasets.mnist
(X_train, y_train), _ = mnist.load_data()
X_train = X_train / 255

X_train = np.expand_dims(X_train, axis=-1)

# create the member, non member, and calibration datasets
member_data = X_train[:amount_points]
member_labels = y_train[:amount_points]
non_member_data = X_train[amount_points:(amount_points*2)]
non_member_labels = y_train[amount_points:(amount_points*2)]
# Use a holdout of the train set to simulate calibration data for the DkNN.
#TODO define calibration data differently?
calibration_data = X_train[(amount_points):((amount_points)+amount_calibration)]
calibration_labels = y_train[(amount_points):(amount_points)+amount_calibration]

print("Shape of member data: {}".format(member_data.shape))
print("Shape of member labels: {}".format(member_labels.shape))
print("Shape of non member data: {}".format(non_member_data.shape))
print("Shape of non member labels: {}".format(non_member_labels.shape))
print("Shape of calibration data: {}".format(calibration_data.shape))
print("Shape of calibration labels: {}".format(calibration_labels.shape))

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
history = model.fit(
    member_data,
    member_labels,
    epochs=12,
    batch_size=128
)

#model.summary()
train_accuracy = model.evaluate(member_data, member_labels)
test_accuracy = model.evaluate(non_member_data, non_member_labels)
print("Train accuracy: {}".format(train_accuracy[1]))
print("Test accuracy: {}".format(test_accuracy[1]))


# summarize filter shapes per layer
# we are only interested in convolutional layers
#print("Layer names (and shapes) of the model:")
layer_indices = []
nb_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]

    # check for convolutional layer
    if ("conv" not in layer.name) and ("dense" not in layer.name):
        #print(layer.name)
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    #print(layer.name, filters.shape)
    nb_layers.append(layer.name)
    layer_indices.append(i)

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

all_data_one_experiment_for_pickle = []
experiment_data_for_pickle = []
counter = 0
for scale, k_neighbors, amount_generate_neighbors in HYPERPARAMETERS:
    print("experiment number: {}".format(counter))
    counter += 1
    for element in range(AMOUNT_M_NM_TOTAL):
        print("element number: {}".format(element))
        # Wrap the model into a DkNNModel
        dknn = DkNNModel(
            neighbors=k_neighbors,
            layers=nb_layers,
            get_activations=get_activations,
            train_data=member_data,
            train_labels=member_labels,
            back_end= backend,
        )

        print("Start model calibration")
        dknn.calibrate(calibration_data, calibration_labels)
        print("Calibrated the model")

        #generate neighbors
        #for member data
        generated_data_member = generate_neighboring_points(member_data[element], amount=amount_generate_neighbors, scale=scale)
        data_fprop_member = np.concatenate(([member_data[element]], generated_data_member), axis = 0)
        #for non member data
        generated_data_non_member = generate_neighboring_points(non_member_data[element], amount=amount_generate_neighbors, scale=scale)
        data_fprop_non_member = np.concatenate(([non_member_data[element]], generated_data_non_member), axis = 0)
        # forward propagation through DkNN
        _, knns_ind_member, _, knns_distances_member = dknn.fprop_np(data_fprop_member, get_knns=True)
        _, knns_ind_non_member, _, knns_distances_non_member = dknn.fprop_np(data_fprop_non_member, get_knns=True)

        current_data_for_pickle = []
        current_data_for_pickle.append(train_accuracy)
        current_data_for_pickle.append(test_accuracy)
        current_data_for_pickle.append(knns_ind_member)
        current_data_for_pickle.append(knns_ind_non_member)
        current_data_for_pickle.append(knns_distances_member)
        current_data_for_pickle.append(knns_distances_non_member)
        current_data_for_pickle.append(scale)
        current_data_for_pickle.append(k_neighbors)
        current_data_for_pickle.append(amount_generate_neighbors)
        current_data_for_pickle.append(HYPERPARAMETERS)
        current_data_for_pickle.append(AMOUNT_M_NM_TOTAL)

        all_data_one_experiment_for_pickle.append(current_data_for_pickle)

    experiment_data_for_pickle.append(all_data_one_experiment_for_pickle)
    all_data_one_experiment_for_pickle = []
    with open('/home/inafen/jupyter_notebooks/data_neighbors_test2.pickle', 'wb') as f:
        pickle.dump(experiment_data_for_pickle, f)


print("--- %s seconds ---" % (time.time() - start_time))
