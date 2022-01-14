import numpy as np
from collections import Counter, defaultdict


def get_differences_knns_btw_layers(data, knns_attribute, num_instead_of_data = False, compares_with_first_layer_only = False):
    """
    Looks at an attribute of the k nearest neighbors of data for each layer (DkNN) and outputs the changes of the knns between the layers.

    :param data: the data for which the knns were found, in other words: were forwarded together through DkNN
    :param knns_attribute: indices or labels of knns
    :return: changed_knns_all_points (e.g. point 2: {layer 1: [11][12]}) , differences_knns_all_points (e.g. point 2: {layer 1: 2}), differences_knns_sum (e.g. layer 2: [2,3,1])
    """
    #TODO add description num instead of data
    differences_knns_point = {}
    changed_knns_point = {}
    differences_knns_all_points = {}
    changed_knns_all_points = {}
    differences_knns_total = {}

    if num_instead_of_data:
        amount_data = data
    else:
        amount_data = len(data)
    for data_point in range(amount_data):
        for layer in range(len(knns_attribute)):
            if layer == 0:
                knns_current_layer_point = knns_attribute[layer][1][data_point]
            else:
                knns_next_layer_point = knns_attribute[layer][1][data_point]
                # compare whether same elements in nn_current_layer and nn_next_layer
                if Counter(knns_current_layer_point) == Counter(knns_next_layer_point):
                    # save amount of changes for all points to be able to calculate mean later
                    if "layer {}".format(layer) in differences_knns_total:
                        differences_knns_total["layer {}".format(layer)].append(0)
                    else:
                        differences_knns_total["layer {}".format(layer)] = [0]
                else:
                    # save amount of changes (differences_knns_point), which knns changed (changed_knns_point)
                    changed_knns = list(
                        set(knns_current_layer_point) - set(knns_next_layer_point)
                    )
                    differences_knns_point["layer {}".format(layer)] = len(changed_knns)
                    changed_knns_point["layer {}".format(layer)] = (
                        changed_knns,
                        (
                            list(
                                set(knns_next_layer_point)
                                - set(knns_current_layer_point)
                            )
                        ),
                    )

                    # save amount of changes for all points to be able to calculate mean later
                    #TODO never goes in if
                    if "layer {}".format(layer) in differences_knns_total:
                        differences_knns_total["layer {}".format(layer)].append(
                            len(changed_knns)
                        )
                    else:
                        differences_knns_total["layer {}".format(layer)] = [
                            (len(changed_knns))
                        ]
                if compares_with_first_layer_only == False:
                        #if yes, changes btw two layers: the former next layer becomes the current layer
                    knns_current_layer_point = knns_attribute[layer][1][data_point]
        differences_knns_all_points[
            "point {}".format(data_point)
        ] = differences_knns_point
        changed_knns_all_points["point {}".format(data_point)] = changed_knns_point
    return changed_knns_all_points, differences_knns_all_points, differences_knns_total


def get_mean_knns_layer(knns_attribute, differences_knns_total):
    """
    Get the mean of an attribute of the knns per inbetween two layers, e.g. mean of changes of knns btw. two layers

    :param knns_attribute: indices or labels of knns
    :param differences_knns_total: how many changes in knns have happened between two layers
    :return: mean_knns_layers
    """

    # get mean of knns per layer
    mean_knns_layers = []
    for layer in range(1, len(knns_attribute)):
        mean_knns_layers.append(
            np.mean(differences_knns_total.get("layer {}".format(layer)))
        )
    return mean_knns_layers


def get_distances_of_knns(data, knns_distances, num_instead_of_data):
    """
    Put distances of all knns in an dict with layers as keys

    :param data: data
    :param knns_distances: distances of knns
    :return: distances_knns_all
    """
    distances_knns_all = {}

    if num_instead_of_data:
        amount_data = data
    else:
        amount_data = len(data)
    for data_point in range(amount_data):
        for layer in range(len(knns_distances)):

            if "layer {}".format(layer) in distances_knns_all:
                distances_knns_all["layer {}".format(layer)].append(
                    knns_distances[layer][1][data_point]
                )
            else:
                distances_knns_all["layer {}".format(layer)] = [
                    knns_distances[layer][1][data_point]
                ]
    return distances_knns_all


def get_mean_distances_of_knns(distances_knns_all:dict, knns_distances):
    """
    Get the mean distance of knns for one layer.

    :param distances_knns_all: distance of all knns as dict with layers as keys
    :param knns_distances: distances of knns as list
    :return: mean_distances_knns_layers
    """
    mean_distances_knns_layers = []
    for layer in range(0, len(knns_distances)):
        mean_distances_knns_layers.append(
            np.mean(distances_knns_all.get("layer {}".format(layer)))
        )
    return mean_distances_knns_layers

def get_sum_similarities_of_knns(similarities_knns_all, knns_similarities):
    sum_similarity_knns_layers = []
    for layer in range(1, len(knns_similarities)):
        sum_similarity_knns_layers.append(
            np.sum(similarities_knns_all.get("layer {}".format(layer)))
        )
    return sum_similarity_knns_layers

def get_similarities_knns_btw_layers(data, knns_attribute, num_instead_of_data = False, compares_with_first_layer_only = False):
    #TODO change names
    """
    Looks at an attribute of the k nearest neighbors of data for each layer (DkNN) and outputs the changes of the knns between the layers.

    :param data: the data for which the knns were found, in other words: were forwarded together through DkNN
    :param knns_attribute: indices or labels of knns

    :param compares_with_first_layer_only: When True, it is checked which knns stay the same as in the first layer (e.g. layer 0 [1], layer 1 [2], layer 2 [2] --> no similarity),
                                       otherwise only whether they stay the same as in the last layer (e.g. layer 0 [1], layer 1 [2], layer 2 [2] --> one similarity)
    :return: changed_knns_all_points (e.g. point 2: {layer 1: [11][12]}) , differences_knns_all_points (e.g. point 2: {layer 1: 2}), differences_knns_sum (e.g. layer 2: [2,3,1])
    """
    #TODO add description num instead of data
    similarities_knns_point = {}
    similar_knns_point = {}
    similarities_knns_all_points = {}
    similar_knns_all_points = {}
    similarities_knns_total = {}

    if num_instead_of_data:
        amount_data = data
    else:
        amount_data = len(data)
    for data_point in range(amount_data):
        for layer in range(len(knns_attribute)):
            if layer == 0:
                knns_current_layer_point = knns_attribute[layer][1][data_point]
            else:
                knns_next_layer_point = knns_attribute[layer][1][data_point]
                # compare whether same elements in nn_current_layer and nn_next_layer
                if Counter(knns_current_layer_point) == Counter(knns_next_layer_point):
                    # save amount of similarities for all points
                    if "layer {}".format(layer) in similarities_knns_total:
                        similarities_knns_total["layer {}".format(layer)].append(len(knns_current_layer_point))
                    else:
                        similarities_knns_total["layer {}".format(layer)] = [len(knns_current_layer_point)]
                else:
                    # save which knns stay similar in current and next layer
                    similar_knns = list(
                        set(knns_current_layer_point) & set(knns_next_layer_point)
                    )
                    #get the amount of knns that stay similar for one point
                    similarities_knns_point["layer {}".format(layer)] = (len(similar_knns))
                    #get the amount of knns that stay similar and the knn indexes themselves for one point
                    similar_knns_point["layer {}".format(layer)] = (
                        similar_knns,
                        (
                            list(
                                set(knns_next_layer_point)
                                & set(knns_current_layer_point)
                            )
                        ),
                    )

                    # save amount of similar knns for all points to be able to easily calculate mean later (or other calculation)
                    if "layer {}".format(layer) in similarities_knns_total:
                        similarities_knns_total["layer {}".format(layer)].append(
                            len(similar_knns)
                        )
                    else:
                        similarities_knns_total["layer {}".format(layer)] = [
                            (len(similar_knns))
                        ]
                if compares_with_first_layer_only == False:
                        #if yes, changes btw two layers: the former next layer becomes the current layer
                    knns_current_layer_point = knns_attribute[layer][1][data_point]

        similarities_knns_all_points[
            "point {}".format(data_point)
        ] = similarities_knns_point
        similar_knns_all_points["point {}".format(data_point)] = similar_knns_point
    return similar_knns_all_points, similarities_knns_all_points, similarities_knns_total