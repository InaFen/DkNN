import numpy as np
from collections import Counter


def get_differences_knns_btw_layers(data, knns_attribute):
    """
    Looks at the indices of the k nearest neighbors of data for each layer (DkNN) and outputs the changes of the knns between the layers.

    :param data: the data for which the knns were found.
    :param knns_attribute: indices or labels of knns
    :return: changed_knns_all_points (e.g. point 2: {layer 1: [11][12]}) , differences_knns_all_points (e.g. point 2: {layer 1: 2}), differences_knns_sum (e.g. layer 2: [2,3,1])
    """
    differences_knns_point = {}
    changed_knns_point = {}
    differences_knns_all_points = {}
    changed_knns_all_points = {}
    differences_knns_total = {}

    for data_point in range(len(data)):
        for layer in range(len(knns_attribute)):
            if layer == 0:
                knns_current_layer_point = knns_attribute[layer][1][data_point]
            else:
                knns_next_layer_point = knns_attribute[layer][1][data_point]
                # compare whether same elements in nn_current_layer and nn_next_layer
                if Counter(knns_current_layer_point) == Counter(knns_next_layer_point):
                    knns_current_layer_point = knns_next_layer_point

                    # save amount of changes for all points to be able to calculate mean later
                    if "layer {}".format(layer) in differences_knns_total:
                        differences_knns_total["layer {}".format(layer)].append(0)
                    else:
                        differences_knns_total["layer {}".format(layer)] = [0]
                else:
                    #save amount of changes (differences_knns_point), which knns changed (changed_knns_point)
                    changed_knns = list(set(knns_current_layer_point) - set(knns_next_layer_point))
                    differences_knns_point["layer {}".format(layer)] = len(changed_knns)
                    changed_knns_point["layer {}".format(layer)] = (
                    changed_knns, (list(set(knns_next_layer_point) - set(knns_current_layer_point))))

                    #save amount of changes for all points to be able to calculate mean later
                    if "layer {}".format(layer) in differences_knns_total:
                        differences_knns_total["layer {}".format(layer)].append(len(changed_knns))
                    else:
                        differences_knns_total["layer {}".format(layer)] = [(len(changed_knns))]

        differences_knns_all_points["point {}".format(data_point)] = differences_knns_point
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
        mean_knns_layers.append(np.mean(differences_knns_total.get("layer {}".format(layer))))
    return mean_knns_layers


def get_distances_of_knns(data, knns_distances):
    """
    Put distances of all knns in an dict with layers as keys

    :param data: data
    :param knns_distances: distances of knns
    :return: distances_knns_all
    """
    distances_knns_all = {}

    for data_point in range(len(data)):
        for layer in range(len(knns_distances)):

            if "layer {}".format(layer) in distances_knns_all:
                distances_knns_all["layer {}".format(layer)].append(knns_distances[layer][1][data_point])
            else:
                distances_knns_all["layer {}".format(layer)] = [knns_distances[layer][1][data_point]]
    return distances_knns_all


def get_mean_distances_of_knns(distances_knns_all, knns_distances):
    """
    Get the mean distance of knns for one layer.

    :param distances_knns_all: distance of all knns as dict with layers as keys
    :param knns_distances: distances of knns as list
    :return: mean_distances_knns_layers
    """
    mean_distances_knns_layers = []
    for layer in range(1, len(knns_distances)):
        mean_distances_knns_layers.append(np.mean(distances_knns_all.get("layer {}".format(layer))))
    return mean_distances_knns_layers





