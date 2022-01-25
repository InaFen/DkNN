from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dknn import *
from utils.utils_model import *
from utils.utils_neighbors import generate_neighboring_points
from utils.utils_plot import plot_images

from matplotlib import pyplot as plt
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

K_NEIGHBORS = 10  # [5, 10, 50, 100]  # for DkNN
AMOUNT_GENERATE_NEIGHBORS = 200  # [50, 100, 200]  # for generate_neighboring_points
SCALES_EPSILON = [
    (0.005, 0.2),
    (0.01, 0.3),
    (0.02, 0.4),
    (0.05, 0.7),
    (0.1, 1.5),
    (0.2, 3.0),
]
AMOUNT_DATA_TOTAL = 100  # how many elements are used

# parameters for testing
amount_points = 30000
amount_calibration = 10
backend = NearestNeighbor.BACKEND.FAISS
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


def experiments_setup_DkNN(
    train_data_DkNN,
    train_labels_DkNN,
    calibration_data,
    calibration_label,
    data_fprop_DkNN,
    labels_fprop_DkNN,
    filepath_pickle="/home/inafen/jupyter_notebooks/temp.pickle",
    get_activations=get_activations,
    k_neighbors=K_NEIGHBORS,
    backend=backend,
    nb_layers=nb_layers,
    save_pickle=True,
    amount_data_total=AMOUNT_DATA_TOTAL,
):

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
        current_data_for_pickle["amount_generate_neighbors"] = AMOUNT_GENERATE_NEIGHBORS
        current_data_for_pickle["AMOUNT_M_NM_TOTAL"] = amount_data_total
        all_data_one_experiment_for_pickle[
            "element {}".format(element)
        ] = current_data_for_pickle

    if save_pickle == True:
        with open(filepath_pickle, "wb") as f:
            pickle.dump(all_data_one_experiment_for_pickle, f)
    else:
        return all_data_one_experiment_for_pickle


# experiment 1.1
experiments_setup_DkNN(
    member_data,
    member_labels,
    non_member_data[:amount_calibration],
    non_member_labels[:amount_calibration],
    non_member_data[amount_calibration : amount_calibration + 100],
    non_member_labels[amount_calibration : amount_calibration + 100],
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_1.pickle",
)

# experiment 1.2
# same set up as 1.1 but one noisy and one not noisy element is passed forward through DkNN at the same time
# generate noisy data point
# TODO very similar to 3.1, so if wanted merge together, rename
noisy_point_3_1 = generate_neighboring_points(
    non_member_data[0], 1, scale=0.9, epsilon=9
)[0]
noisy_point_label_3_1 = np.array(non_member_labels[0], dtype=np.uint8)

labels_fprop_element_1_2 = []
all_data_experiment_1_2_for_pickle = {}
for element in range(AMOUNT_DATA_TOTAL):
    data_fprop_element_1_2 = np.concatenate(
        (
            np.expand_dims(noisy_point_3_1, axis=0),
            np.expand_dims(non_member_data[element], axis=0),
        )
    )
    labels_fprop_element_1_2.extend((noisy_point_label_3_1, non_member_labels[element]))
    all_data_experiment_1_2_for_pickle[element] = experiments_setup_DkNN(
        train_data_DkNN=member_data,
        train_labels_DkNN=member_labels,
        calibration_data=non_member_data[
            AMOUNT_DATA_TOTAL : (AMOUNT_DATA_TOTAL + amount_calibration)
        ],
        calibration_label=non_member_labels[
            AMOUNT_DATA_TOTAL : (AMOUNT_DATA_TOTAL + amount_calibration)
        ],
        filepath_pickle=None,
        data_fprop_DkNN=data_fprop_element_1_2,
        labels_fprop_DkNN=labels_fprop_element_1_2,
        save_pickle=False,
        amount_data_total=2,
    )
    with open(
        "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_2.pickle", "wb"
    ) as f:
        pickle.dump(all_data_experiment_1_2_for_pickle, f)

# experiment 2.1
# generate neighbors to have data with much noise
generated_neighbors = np.zeros(
    (AMOUNT_GENERATE_NEIGHBORS + amount_calibration, 28, 28, 1)
)
generated_neighbors_labels = np.zeros(
    AMOUNT_GENERATE_NEIGHBORS + amount_calibration, dtype=np.uint8
)
for index in range(AMOUNT_GENERATE_NEIGHBORS + amount_calibration):
    generated_neighbors[index] = generate_neighboring_points(
        non_member_data[index], 1, scale=0.9, epsilon=9
    )  # generate 1 neighbor for each point
    generated_neighbors_labels[index] = non_member_labels[index]
# Print images to decide on suiting scale and epsilon

images = generated_neighbors[:6]
_, axs = plt.subplots(1, 5, figsize=(8, 8))
axs = axs.flatten()
for image, ax in zip(images, axs):
    ax.imshow(image.squeeze(), cmap="gray")
plt.show()
plt.savefig(
    "/home/inafen/jupyter_notebooks/generated_neighbors_noise.png", bbox_inches="tight"
)
plt.clf()

amount_no_noise_data_element = 4
no_noise_data_element = np.zeros((amount_no_noise_data_element, 28, 28, 1))
no_noise_labels_element = np.zeros((amount_no_noise_data_element))

mixed_noise_no_noise_data = np.zeros(
    (
        AMOUNT_DATA_TOTAL,
        amount_no_noise_data_element + AMOUNT_GENERATE_NEIGHBORS,
        28,
        28,
        1,
    )
)
mixed_noise_no_noise_labels = np.zeros(
    (AMOUNT_DATA_TOTAL, amount_no_noise_data_element + AMOUNT_GENERATE_NEIGHBORS)
)

all_data_experiment_2_1_for_pickle = {}
# since every element will have its own DkNN training data, go through every element individually
for element in range(AMOUNT_DATA_TOTAL):
    counter = 0
    # look through data that will not be input in fprop to find data with same label for DkNN train data
    non_member_labels_no_fprop = non_member_labels[AMOUNT_DATA_TOTAL:]
    non_member_data_no_fprop = non_member_data[AMOUNT_DATA_TOTAL:]
    for index in range(len(non_member_labels_no_fprop)):
        # break if amount_no_noise_data_element elements are found
        if counter == (amount_no_noise_data_element):
            break
        if non_member_labels[element] == non_member_labels_no_fprop[index]:
            no_noise_data_element[counter] = non_member_data_no_fprop[index]
            no_noise_labels_element[counter] = non_member_labels_no_fprop[index]
            counter += 1
    # prepare data with noisy data and non noisy data for experiment 2.1 for each element
    mixed_noise_no_noise_data[element] = np.concatenate(
        (no_noise_data_element, generated_neighbors[:AMOUNT_GENERATE_NEIGHBORS])
    )
    mixed_noise_no_noise_labels[element] = np.concatenate(
        (
            no_noise_labels_element,
            generated_neighbors_labels[:AMOUNT_GENERATE_NEIGHBORS],
        )
    )
    # since every element will have its own DkNN training data, the experiment has to be done individually
    # caution: it is element:element: instead of element:
    all_data_experiment_2_1_for_pickle[element] = experiments_setup_DkNN(
        train_data_DkNN=mixed_noise_no_noise_data[element],
        train_labels_DkNN=mixed_noise_no_noise_labels[element],
        calibration_data=generated_neighbors[AMOUNT_GENERATE_NEIGHBORS:],
        calibration_label=generated_neighbors_labels[AMOUNT_GENERATE_NEIGHBORS:],
        filepath_pickle=None,
        data_fprop_DkNN=non_member_data[element],
        labels_fprop_DkNN=non_member_labels[element],
        save_pickle=False,
        amount_data_total=1,
    )
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_2_1.pickle", "wb"
) as f:
    pickle.dump(all_data_experiment_2_1_for_pickle, f)


# experiment 3.1
# DkNN gets trained with generated neighbors
# generate individual neighbors for each point
generated_neighbors_3_1 = np.zeros(
    (AMOUNT_DATA_TOTAL, AMOUNT_GENERATE_NEIGHBORS + amount_calibration, 28, 28, 1)
)
generated_neighbors_labels_3_1 = np.zeros(
    (AMOUNT_DATA_TOTAL, AMOUNT_GENERATE_NEIGHBORS + amount_calibration), dtype=np.uint8
)
for element in range(AMOUNT_DATA_TOTAL):
    generated_neighbors_3_1[element] = generate_neighboring_points(
        non_member_data[element],
        AMOUNT_GENERATE_NEIGHBORS + amount_calibration,
        scale=0.2,
        epsilon=3,
    )
    generated_neighbors_labels_3_1[element] = np.full(
        (AMOUNT_GENERATE_NEIGHBORS + amount_calibration), non_member_labels[element]
    )

# generate noisy data point
noisy_point_3_1 = generate_neighboring_points(
    non_member_data[0], 1, scale=0.9, epsilon=9
)[0]
noisy_point_label_3_1 = np.array(non_member_labels[0], dtype=np.uint8)
print(noisy_point_label_3_1)

# Print images to decide on suiting scale and epsilon
images = generated_neighbors_3_1[0][:6]
_, axs = plt.subplots(1, 5, figsize=(8, 8))
axs = axs.flatten()
for image, ax in zip(images, axs):
    ax.imshow(image.squeeze(), cmap="gray")
plt.show()
plt.savefig(
    "/home/inafen/jupyter_notebooks/generated_neighbors_3_1.png", bbox_inches="tight"
)
plt.clf()

all_data_experiment_3_1_for_pickle = {}
labels_fprop_element = []
# every element has individual neighbors so individual fprop
for element in range(AMOUNT_DATA_TOTAL):
    # send one noisy and element through DkNN
    # expand dimension so data_fprop_element has correct dimension: (2,28,28,1) instead of (56,28,1)
    data_fprop_element = np.concatenate(
        (
            np.expand_dims(noisy_point_3_1, axis=0),
            np.expand_dims(non_member_data[element], axis=0),
        )
    )
    labels_fprop_element.extend((noisy_point_label_3_1, non_member_labels[element]))
    all_data_experiment_3_1_for_pickle[element] = experiments_setup_DkNN(
        train_data_DkNN=generated_neighbors_3_1[element][:AMOUNT_GENERATE_NEIGHBORS],
        train_labels_DkNN=generated_neighbors_labels_3_1[element][
            :AMOUNT_GENERATE_NEIGHBORS
        ],
        calibration_data=generated_neighbors_3_1[element][AMOUNT_GENERATE_NEIGHBORS:],
        calibration_label=generated_neighbors_labels_3_1[element][
            AMOUNT_GENERATE_NEIGHBORS:
        ],
        filepath_pickle=None,
        data_fprop_DkNN=data_fprop_element,
        labels_fprop_DkNN=labels_fprop_element,
        save_pickle=False,
        amount_data_total=2,
    )
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_3_1.pickle", "wb"
) as f:
    pickle.dump(all_data_experiment_3_1_for_pickle, f)


# experiment 4.1
# DkNN gets trained with little suiting generated neighbors and otherwise noisy data (similar to 2.1)
# TODO same as 2.1 so if wanted merge, rename later
# generate neighbors to have data with much noise
generated_neighbors = np.zeros(
    (AMOUNT_GENERATE_NEIGHBORS + amount_calibration, 28, 28, 1)
)
generated_neighbors_labels = np.zeros(
    AMOUNT_GENERATE_NEIGHBORS + amount_calibration, dtype=np.uint8
)
for index in range(AMOUNT_GENERATE_NEIGHBORS + amount_calibration):
    generated_neighbors[index] = generate_neighboring_points(
        non_member_data[index], 1, scale=0.9, epsilon=9
    )  # generate 1 neighbor for each point
    generated_neighbors_labels[index] = non_member_labels[index]

amount_no_noise_data_element = 4
not_noisy_neighbors_element_4_1 = np.zeros((amount_no_noise_data_element, 28, 28, 1))
not_noisy_neighbors_element_labels_4_1 = np.zeros((amount_no_noise_data_element))

mixed_noise_no_noise_data_neighbors_4_1 = np.zeros(
    (
        AMOUNT_DATA_TOTAL,
        amount_no_noise_data_element + AMOUNT_GENERATE_NEIGHBORS,
        28,
        28,
        1,
    )
)
mixed_noise_no_noise_labels_neighbors_4_1 = np.zeros(
    (AMOUNT_DATA_TOTAL, amount_no_noise_data_element + AMOUNT_GENERATE_NEIGHBORS)
)

all_data_experiment_4_1_for_pickle = {}
# since every element will have its own DkNN training data, go through every element individually
for element in range(AMOUNT_DATA_TOTAL):
    counter = 0
    # create neighbors that are similar to data point
    not_noisy_neighbors_element_4_1 = generate_neighboring_points(
        non_member_data[element],
        amount=amount_no_noise_data_element,
        scale=0.2,
        epsilon=3,
    )
    not_noisy_neighbors_element_labels_4_1 = np.full(
        amount_no_noise_data_element, non_member_labels[element]
    )

    # prepare data with noisy data and non noisy data for experiment 2.1 for each element
    mixed_noise_no_noise_data_neighbors_4_1[element] = np.concatenate(
        (
            not_noisy_neighbors_element_4_1,
            generated_neighbors[:AMOUNT_GENERATE_NEIGHBORS],
        )
    )
    mixed_noise_no_noise_labels_neighbors_4_1[element] = np.concatenate(
        (
            not_noisy_neighbors_element_labels_4_1,
            generated_neighbors_labels[:AMOUNT_GENERATE_NEIGHBORS],
        )
    )
    # since every element will have its own DkNN training data, the experiment has to be done individually
    # caution: it is element:element: instead of element:
    all_data_experiment_4_1_for_pickle[element] = experiments_setup_DkNN(
        train_data_DkNN=mixed_noise_no_noise_data_neighbors_4_1[element],
        train_labels_DkNN=mixed_noise_no_noise_labels_neighbors_4_1[element],
        calibration_data=generated_neighbors[AMOUNT_GENERATE_NEIGHBORS:],
        calibration_label=generated_neighbors_labels[AMOUNT_GENERATE_NEIGHBORS:],
        filepath_pickle=None,
        data_fprop_DkNN=non_member_data[element],
        labels_fprop_DkNN=non_member_labels[element],
        save_pickle=False,
        amount_data_total=1,
    )
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_4_1.pickle", "wb"
) as f:
    pickle.dump(all_data_experiment_4_1_for_pickle, f)

print("--- %s seconds ---" % (time.time() - start_time))
