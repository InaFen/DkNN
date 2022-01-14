from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils.utils_model import *
from utils.utils_neighbors import *


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

K_NEIGHBORS = [5, 10, 50, 100]  # for DkNN
AMOUNT_GENERATE_NEIGHBORS = [50, 100, 200]  # for generate_neighboring_points
SCALES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]  # for generate_neighboring_points
# how many of member/non member elements are used
AMOUNT_M_NM_TOTAL = 1000

# parameters for testing
amount_points = 30000
backend = (
    NearestNeighbor.BACKEND.FAISS
)  # TODO FALCONN does not work for small amount of data, most likely because data is in different buckets (LSH) --> if wanted, an alternative has to be implemented
path_model = "/home/inafen/jupyter_notebooks/model_lnet5_2"

hyperparamters = np.array((SCALES, K_NEIGHBORS, AMOUNT_GENERATE_NEIGHBORS))
HYPERPARAMETERS = list(itertools.product(*hyperparamters))  # all experiments
print(HYPERPARAMETERS)

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


# generate neighbors for member, non-member data
# at max, 200 neighbors need to be generated (see AMOUNT_GENERATE_NEIGHBORS), so generate 200 neighbors for each point
# for all different scales
max_amount_generate_neighbors = max(AMOUNT_GENERATE_NEIGHBORS)
amount_calibration = 10


# [scales[amount (non)members[amount neighbors[28[ 28 [1]]]]]]
generated_neighbors_per_scale_member_all = np.zeros(
    shape=(
        len(SCALES),
        AMOUNT_M_NM_TOTAL,
        (max_amount_generate_neighbors + amount_calibration),
        28,
        28,
        1,
    )
)
generated_neighbors_per_scale_non_member_all = np.zeros(
    shape=(
        len(SCALES),
        AMOUNT_M_NM_TOTAL,
        (max_amount_generate_neighbors + amount_calibration),
        28,
        28,
        1,
    )
)

# Use a holdout of the generated neighbors to simulate calibration data for the DkNN.
calibration_data_per_scale_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, amount_calibration, 28, 28, 1)
)
calibration_data_per_scale_non_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, amount_calibration, 28, 28, 1)
)

calibration_label_per_scale_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, amount_calibration), dtype=np.uint8
)
calibration_label_per_scale_non_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, amount_calibration), dtype=np.uint8
)

generated_neighbors_per_scale_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, max_amount_generate_neighbors, 28, 28, 1)
)
generated_neighbors_per_scale_non_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, max_amount_generate_neighbors, 28, 28, 1)
)

label_generated_neighbors_per_scale_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, max_amount_generate_neighbors),
    dtype=np.uint8,
)
label_generated_neighbors_per_scale_non_member = np.zeros(
    shape=(len(SCALES), AMOUNT_M_NM_TOTAL, max_amount_generate_neighbors),
    dtype=np.uint8,
)

calibration_data = X_train[(amount_points) : ((amount_points) + amount_calibration)]
calibration_labels = y_train[(amount_points) : (amount_points) + amount_calibration]
print(calibration_data.shape)

for scale_ind in range(len(SCALES)):
    for element in range(AMOUNT_M_NM_TOTAL):
        generated_neighbors_per_scale_member_all[scale_ind][
            element
        ] = generate_neighboring_points(
            member_data[element],
            amount=(max_amount_generate_neighbors + amount_calibration),
            scale=SCALES[scale_ind],
        )
        calibration_data_per_scale_member[scale_ind][
            element
        ] = generated_neighbors_per_scale_member_all[scale_ind][element][
            :amount_calibration
        ]
        generated_neighbors_per_scale_member[scale_ind][
            element
        ] = generated_neighbors_per_scale_member_all[scale_ind][element][
            amount_calibration:
        ]

        label_generated_neighbors_per_scale_member[scale_ind][element] = np.full(
            (max_amount_generate_neighbors), member_labels[element]
        )
        label_generated_neighbors_per_scale_member[scale_ind][element] = np.full(
            (max_amount_generate_neighbors), member_labels[element]
        )

        generated_neighbors_per_scale_non_member_all[scale_ind][
            element
        ] = generate_neighboring_points(
            non_member_data[element],
            amount=(max_amount_generate_neighbors + amount_calibration),
            scale=SCALES[scale_ind],
        )
        calibration_data_per_scale_non_member[scale_ind][
            element
        ] = generated_neighbors_per_scale_non_member_all[scale_ind][element][
            :amount_calibration
        ]
        generated_neighbors_per_scale_member[scale_ind][
            element
        ] = generated_neighbors_per_scale_non_member_all[scale_ind][element][
            amount_calibration:
        ]

        calibration_label_per_scale_non_member[scale_ind][element] = np.full(
            (amount_calibration), non_member_labels[element]
        )
        label_generated_neighbors_per_scale_non_member[scale_ind][element] = np.full(
            (max_amount_generate_neighbors), non_member_labels[element]
        )

all_data_one_experiment_for_pickle = []
experiment_data_for_pickle = []
counter = 0
for scale, k_neighbors, amount_generate_neighbors in HYPERPARAMETERS:
    print("experiment number: {}".format(counter))
    counter += 1
    # get index of scale
    for i in range(len(SCALES)):
        if scale == SCALES[i]:
            scale_index = i
            break
    for element in range(AMOUNT_M_NM_TOTAL):
        print("experiment number: {}".format(counter))
        print("element number: {}".format(element))

        # Wrap the model into a DkNNModel
        # DkNN for Member datapoint
        dknn_member = DkNNModel(
            neighbors=k_neighbors,
            layers=nb_layers,
            get_activations=get_activations,
            train_data=generated_neighbors_per_scale_member[scale_index][element][
                :amount_generate_neighbors
            ],
            train_labels=label_generated_neighbors_per_scale_member[scale_ind][element][
                :amount_generate_neighbors
            ],
            back_end=backend,
        )

        # DkNN for Non-Member datapoint

        dknn_non_member = DkNNModel(
            neighbors=k_neighbors,
            layers=nb_layers,
            get_activations=get_activations,
            train_data=generated_neighbors_per_scale_non_member[scale_index][element][
                :amount_generate_neighbors
            ],
            train_labels=label_generated_neighbors_per_scale_non_member[scale_ind][
                element
            ][:amount_generate_neighbors],
            back_end=backend,
        )

        cal_data_temp = calibration_data_per_scale_member[scale_index][element][
            :amount_calibration
        ]
        cal_label_temp = calibration_label_per_scale_member[scale_index][element]
        cal_label_temp_shape = cal_label_temp.shape[0]
        cal_label_temp_shape = cal_label_temp.shape[0]

        # calibrate models
        # dknn_member.calibrate(generated_neighbors_per_scale_member[scale_index][element][:amount_calibration],
        #                      label_generated_neighbors_per_scale_member[scale_index][element][:amount_calibration])
        dknn_member.calibrate(
            calibration_data_per_scale_member[scale_index][element],
            calibration_label_per_scale_member[scale_index][element],
        )

        dknn_non_member.calibrate(
            calibration_data_per_scale_non_member[scale_index][element][
                :amount_calibration
            ],
            calibration_label_per_scale_non_member[scale_index][element][
                :amount_calibration
            ],
        )

        # forward propagation through DkNNs
        _, knns_ind_member, _, knns_distances_member = dknn_member.fprop_np(
            member_data[element], get_knns=True
        )
        _, knns_ind_non_member, _, knns_distances_non_member = dknn_non_member.fprop_np(
            non_member_data[element], get_knns=True
        )

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
    with open("/home/inafen/jupyter_notebooks/data_neighbors_test6.pickle", "wb") as f:
        pickle.dump(experiment_data_for_pickle, f)


print("--- %s seconds ---" % (time.time() - start_time))
