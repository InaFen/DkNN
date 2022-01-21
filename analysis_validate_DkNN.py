import pickle
from utils.utils_plot import plot_images
import numpy as np

# experiment 1.1
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_1.pickle", "rb"
) as f:
    loaded_obj = pickle.load(f)

# get unsimilar labels
print(
    "Unsimilar lables of knns to elements per layer (= [] means all knn labels == element label in this layer)"
)
for element in range(loaded_obj["element 0"]["AMOUNT_M_NM_TOTAL"]):
    unsimilar_labels_to_point = []
    for layer in loaded_obj["element {}".format(element)]["knns_labels"]:
        labels_knn_layer = loaded_obj["element {}".format(element)]["knns_labels"][
            layer
        ][
            0
        ]  # [0] because array has form [[x,x,..]]
        unsimilar_labels_to_point_layer = [
            label
            for label in labels_knn_layer
            if loaded_obj["element {}".format(element)]["label of element"] != label
        ]
        unsimilar_labels_to_point.append(unsimilar_labels_to_point_layer)
    print(
        "Element {}, label {}: {}".format(
            element,
            loaded_obj["element {}".format(element)]["label of element"],
            unsimilar_labels_to_point,
        )
    )

# plot data points for point where knn labels are same as point label
print("Data point and knns where all knn labels are the same over all layers")
data_element = np.expand_dims(loaded_obj["element 2"]["element"], axis=0)
print("The data point:")
plot_images(
    images=data_element,
    amount_subplots=1,
    filepath="/home/inafen/jupyter_notebooks/exp_1_1_no_changes_point.png",
)
data_knns = []
for layer in loaded_obj["element 2"]["knns_ind"]:  # for each layer
    data_knns_layer = []
    for knn_ind_layer in loaded_obj["element 2"]["knns_ind"][layer][
        0
    ]:  # for each knn index
        data_knns_layer.append(
            loaded_obj["element 2"]["data including knns (= train data DkNN)"][
                knn_ind_layer
            ]
        )
    data_knns.append(data_knns_layer)
for layer in range(len(data_knns)):
    print("Knns for layer {}".format(layer))
    print("Model ")  # TODO
    plot_images(
        images=data_knns[layer],
        amount_subplots=loaded_obj["element 2"]["k_neighbors"],
        filepath="/home/inafen/jupyter_notebooks/exp_1_1_no_changes_{}.png".format(
            layer
        ),
    )

# plot data points for point where knn labels are not same as point label 1/2
print("Data point and knns where all knn labels are the not the same over all layers")
print(
    "Element 39, label {}: [[5, 7, 5, 5], [5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]".format(
        loaded_obj["element 39"]["label of element"]
    )
)
data_element = np.expand_dims(loaded_obj["element 39"]["element"], axis=0)
print("The data point:")
plot_images(
    images=data_element,
    amount_subplots=1,
    filepath="/home/inafen/jupyter_notebooks/exp_1_1_changes_point_1.png",
)
data_knns = []
for layer in loaded_obj["element 39"]["knns_ind"]:  # for each layer
    data_knns_layer = []
    for knn_ind_layer in loaded_obj["element 39"]["knns_ind"][layer][
        0
    ]:  # for each knn index
        data_knns_layer.append(
            loaded_obj["element 39"]["data including knns (= train data DkNN)"][
                knn_ind_layer
            ]
        )
    data_knns.append(data_knns_layer)
for layer in range(len(data_knns)):
    print("Knns for layer {}".format(layer))
    print("Model ")  # TODO
    plot_images(
        images=data_knns[layer],
        amount_subplots=loaded_obj["element 39"]["k_neighbors"],
        filepath="/home/inafen/jupyter_notebooks/exp_1_1_changes_{}.png".format(layer),
    )

# plot data points for point where knn labels are not same as point label 2/2
print("Data point and knns where all knn labels are the not the same over all layers")
print(
    "Element 6, label {}: [[9, 9, 9, 9, 9, 9, 9, 7, 9], [9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9], [], []]".format(
        loaded_obj["element 6"]["label of element"]
    )
)
data_element = np.expand_dims(loaded_obj["element 6"]["element"], axis=0)
print("The data point:")
plot_images(
    images=data_element,
    amount_subplots=1,
    filepath="/home/inafen/jupyter_notebooks/exp_1_1_changes_point_2.png",
)
data_knns = []
for layer in loaded_obj["element 6"]["knns_ind"]:  # for each layer
    data_knns_layer = []
    for knn_ind_layer in loaded_obj["element 6"]["knns_ind"][layer][
        0
    ]:  # for each knn index
        data_knns_layer.append(
            loaded_obj["element 6"]["data including knns (= train data DkNN)"][
                knn_ind_layer
            ]
        )
    data_knns.append(data_knns_layer)
for layer in range(len(data_knns)):
    print("Knns for layer {}".format(layer))
    print("Model ")  # TODO
    plot_images(
        images=data_knns[layer],
        amount_subplots=loaded_obj["element 6"]["k_neighbors"],
        filepath="/home/inafen/jupyter_notebooks/exp_1_1_changes_{}.png".format(layer),
    )

# experiment 1.2
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_1_2.pickle", "rb"
) as f:
    loaded_obj_1_2 = pickle.load(f)
# TODO evaluate experiment

# experiment 2.1
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_2_1.pickle", "rb"
) as f:
    loaded_obj_2_1 = pickle.load(f)

# are the no noise data points always part of knns of element?
# get index of no noise data (always first amount_no_noise_data_element (=4) elements)
for element_index in loaded_obj_2_1:  # for the 100 elements
    all_not_noisy_data_in_knns_element = []
    not_noisy_data_in_knns_element = []
    # not noisy data are the first four elements of train data from DkNN (determined in validate_DkNN:     mixed_noise_no_noise_data[element] =np.concatenate((no_noise_data_element, generated_neighbors[:AMOUNT_GENERATE_NEIGHBORS]))
    indexes_not_noisy_data_element = [0, 1, 2, 3]
    for layer in loaded_obj_2_1[element_index]["element 0"]["knns_ind"]:
        knns_points = []
        # get whether all not noisy data is in knns for element in this layer and which not noisy data is part of knn
        for index in loaded_obj_2_1[element_index]["element 0"]["knns_ind"][layer][0]:
            knns_points.append(
                loaded_obj_2_1[element_index]["element 0"][
                    "data including knns (= train data DkNN)"
                ][index]
            )
        all_not_noisy_data_in_knns_element_layer = all(
            not_noisy_point
            in loaded_obj_2_1[element_index]["element 0"]["knns_ind"][layer][0]
            for not_noisy_point in indexes_not_noisy_data_element
        )
        not_noisy_data_in_knns_element_layer = [
            not_noisy_point
            for not_noisy_point in indexes_not_noisy_data_element
            if not_noisy_point
            in loaded_obj_2_1[element_index]["element 0"]["knns_ind"][layer][0]
        ]
        not_noisy_data_in_knns_element.append(not_noisy_data_in_knns_element_layer)
        all_not_noisy_data_in_knns_element.append(
            all_not_noisy_data_in_knns_element_layer
        )
    print(
        "Element {}, not noisy data NOT all in knns {} times: {}".format(
            element_index,
            len([x for x in all_not_noisy_data_in_knns_element if x == False]),
            all_not_noisy_data_in_knns_element,
        )
    )
    print(
        "Element {}, noisy data, which is in knns per layer: {}".format(
            element_index, not_noisy_data_in_knns_element
        )
    )

# since experiment results are not as expected, lets look at elements with very little not noisy data and very much
# element 92, very different from how expected
print(
    "Let's take a look at: Element 92, noisy data, which is in knns per layer: [[1], [1], [], [], []]"
)
print("Element 92 itself")
plot_images(
    loaded_obj_2_1[92]["element 0"]["element"],
    1,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_92",
)
# the not noisy data
plot_images(
    loaded_obj_2_1[92]["element 0"]["data including knns (= train data DkNN)"][:4],
    4,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_92_not_noisy",
)
# the knns for each layer
for layer in loaded_obj_2_1[92]["element 0"]["knns_ind"]:
    knns_points_92_layer = []
    for index in loaded_obj_2_1[92]["element 0"]["knns_ind"][layer][0]:
        knns_points_92_layer.append(
            loaded_obj_2_1[92]["element 0"]["data including knns (= train data DkNN)"][
                index
            ]
        )
    print("Knns for layer {}".format(layer))
    plot_images(
        knns_points_92_layer,
        10,
        "/home/inafen/jupyter_notebooks/exp_2_1_element_92_knns_layer_{}".format(layer),
    )

# element 97, similar to how expected
print(
    "In contrast, let's take a look at: Element 97, noisy data, which is in knns per layer: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]"
)
print("Element 97 itself")
plot_images(
    loaded_obj_2_1[97]["element 0"]["element"],
    1,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_97",
)
# the not noisy data
plot_images(
    loaded_obj_2_1[97]["element 0"]["data including knns (= train data DkNN)"][:4],
    4,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_97_not_noisy",
)
# the knns for each layer
for layer in loaded_obj_2_1[97]["element 0"]["knns_ind"]:
    knns_points_97_layer = []
    for index in loaded_obj_2_1[97]["element 0"]["knns_ind"][layer][0]:
        knns_points_97_layer.append(
            loaded_obj_2_1[97]["element 0"]["data including knns (= train data DkNN)"][
                index
            ]
        )
    print("Knns for layer {}".format(layer))
    plot_images(
        knns_points_97_layer,
        10,
        "/home/inafen/jupyter_notebooks/exp_2_1_element_97_knns_layer_{}".format(layer),
    )

# experiment 3.1
# similar to 1.2
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_3_1.pickle", "rb"
) as f:
    loaded_obj_3_1 = pickle.load(f)
# TODO evaluate experiment

# experiment 4.1
# similar to 2.1
with open(
    "/home/inafen/jupyter_notebooks/validate_DkNN_experiment_4_1.pickle", "rb"
) as f:
    loaded_obj_4_1 = pickle.load(f)

# are the no noise data points always part of knns of element?
# get index of no noise data (always first amount_no_noise_data_element (=4) elements)
for element_index in loaded_obj_4_1:  # for the 100 elements
    all_not_noisy_data_in_knns_element = []
    not_noisy_data_in_knns_element = []
    # not noisy data are the first four elements of train data from DkNN (determined in validate_DkNN:  mixed_noise_no_noise_data[element] =np.concatenate((no_noise_data_element, generated_neighbors[:AMOUNT_GENERATE_NEIGHBORS]))
    indexes_not_noisy_data_element = [0, 1, 2, 3]
    for layer in loaded_obj_4_1[element_index]["element 0"]["knns_ind"]:
        knns_points = []
        # get whether all not noisy data is in knns for element in this layer and which not noisy data is part of knn
        for index in loaded_obj_4_1[element_index]["element 0"]["knns_ind"][layer][0]:
            knns_points.append(
                loaded_obj_4_1[element_index]["element 0"][
                    "data including knns (= train data DkNN)"
                ][index]
            )
        all_not_noisy_data_in_knns_element_layer = all(
            not_noisy_point
            in loaded_obj_4_1[element_index]["element 0"]["knns_ind"][layer][0]
            for not_noisy_point in indexes_not_noisy_data_element
        )
        not_noisy_data_in_knns_element_layer = [
            not_noisy_point
            for not_noisy_point in indexes_not_noisy_data_element
            if not_noisy_point
            in loaded_obj_4_1[element_index]["element 0"]["knns_ind"][layer][0]
        ]
        not_noisy_data_in_knns_element.append(not_noisy_data_in_knns_element_layer)
        all_not_noisy_data_in_knns_element.append(
            all_not_noisy_data_in_knns_element_layer
        )
    print(
        "Element {}, not noisy data NOT all in knns {} times: {}".format(
            element_index,
            len([x for x in all_not_noisy_data_in_knns_element if x == False]),
            all_not_noisy_data_in_knns_element,
        )
    )
# print("Element {}, noisy data, which is in knns per layer: {}".format(element_index,not_noisy_data_in_knns_element))

# element 97, similar to how expected
print(
    "Let's take a look at: Element 97, noisy data, which is in knns per layer: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]"
)
print("Element 97 itself")
plot_images(
    loaded_obj_4_1[97]["element 0"]["element"],
    1,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_97",
)
# the not noisy data
plot_images(
    loaded_obj_4_1[97]["element 0"]["data including knns (= train data DkNN)"][:4],
    4,
    "/home/inafen/jupyter_notebooks/exp_2_1_element_97_not_noisy",
)
# the knns for each layer
for layer in loaded_obj_4_1[97]["element 0"]["knns_ind"]:
    knns_points_97_4_layer = []
    for index in loaded_obj_4_1[97]["element 0"]["knns_ind"][layer][0]:
        knns_points_97_4_layer.append(
            loaded_obj_4_1[97]["element 0"]["data including knns (= train data DkNN)"][
                index
            ]
        )
    print("Knns for layer {}".format(layer))
    plot_images(
        knns_points_97_4_layer,
        10,
        "/home/inafen/jupyter_notebooks/exp_4_1_element_97_knns_layer_{}".format(layer),
    )

# is the distance of the no noise nn smaller than the other nn?
for element_index in loaded_obj_4_1:  # for the 100 elements
    all_not_noisy_data_in_knns_element = []
    not_noisy_data_in_knns_element = []
    mean_knn_index_distance_not_noisy = []
    mean_knn_index_distance_noisy = []
    # not noisy data are the first four elements of train data from DkNN (determined in validate_DkNN:  mixed_noise_no_noise_data[element] =np.concatenate((no_noise_data_element, generated_neighbors[:AMOUNT_GENERATE_NEIGHBORS]))
    indexes_not_noisy_data_element = [0, 1, 2, 3]
    for layer in loaded_obj_4_1[element_index]["element 0"]["knns_ind"]:
        knn_index_distance_not_noisy = []
        knn_index_distance_noisy = []
        # get the distance of knns, seperated in noisy and not noisy elements to get mean later
        for index in range(
            len(loaded_obj_4_1[element_index]["element 0"]["knns_ind"][layer][0])
        ):
            if (
                loaded_obj_4_1[element_index]["element 0"]["knns_ind"][layer][0][index]
                in indexes_not_noisy_data_element
            ):
                knn_index_distance_not_noisy.append(
                    loaded_obj_4_1[element_index]["element 0"]["knns_distances"][layer][
                        0
                    ][index]
                )
            else:
                knn_index_distance_noisy.append(
                    loaded_obj_4_1[element_index]["element 0"]["knns_distances"][layer][
                        0
                    ][index]
                )
        mean_knn_index_distance_not_noisy.append(np.mean(knn_index_distance_not_noisy))
        mean_knn_index_distance_noisy.append(np.mean(knn_index_distance_noisy))
    print(
        "Element {}, mean distance of not noisy knns: {}, of noisy knns per layer: {}".format(
            element_index,
            mean_knn_index_distance_not_noisy,
            mean_knn_index_distance_noisy,
        )
    )
    print(
        "Element {}, mean distance of not noisy knns < mean distance noisy knns per layer: {}".format(
            element_index,
            [
                True
                for i, j in zip(
                    mean_knn_index_distance_not_noisy, mean_knn_index_distance_noisy
                )
                if i < j
            ],
        )
    )
