#!/usr/bin/env python
# coding: utf-8

# # Plots about DkNN for member & non-members and their generated neighbors

# Each experiment was structered in the following way:
# 1. Similar points/neighbors were generated for all member and non-member points.
# 2. For each member and non-member, the DkNN was trained on the generated neighbors.
# 3. For each member and non-member, the point that the generated neighbors were created on was fed into the DkNN.
# 
# Amount of analysed member/ non-member per experiment: 1000 
# 
# Parameters of experiments:
# 1. K_NEIGHBORS = [5, 10, 50, 100] for DkNN
# 2. AMOUNT_GENERATE_NEIGHBORS = [50, 100, 200] for generate_neighboring_points
# 3. SCALES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# In[1]:


import os
print (os.environ['CONDA_DEFAULT_ENV'])


# In[2]:


import sys
sys.path.append('/home/inafen/jupyter_notebooks/utils/')


# In[13]:


import pickle
from utils.utils import *
from utils.utils_plot import (
    plot_member_non_member_layer_experiments_scatter,
    plot_mean_layer_experiments_all,
    plot_member_non_member_experiments_histogram,
)


# In[8]:


#open pickle
with open('/home/inafen/jupyter_notebooks/data_neighbors_changed_non_member.pickle', 'rb') as f:
    loaded_obj = pickle.load(f)
print("Pickle is loaded")


# In[10]:


#get data from pickle

train_accuracy = []
test_accuracy = []
knns_ind_member = []
knns_ind_non_member = []
knns_distances_member = []
knns_distances_non_member = []
scale = []
k_neighbors = []
amount_generate_neighbors = []
experiment_setups = []
knns_ind_member_points = []
knns_ind_non_member_points = []
knns_distances_member_points = []
knns_distances_non_member_points = []

# get data from pickle
amount_m_nm_total = loaded_obj["experiment 0"]["element 0"]["AMOUNT_M_NM_TOTAL"]
print(amount_m_nm_total)
for experiment in range(len(loaded_obj)):
    for element in range(amount_m_nm_total):
        knns_ind_member_current = []
        knns_ind_non_member_current = []
        knns_distances_member_current = []
        knns_distances_non_member_current = []
        # append values for current point
        knns_ind_member_current.append(
            loaded_obj["experiment {}".format(experiment)][
                "element {}".format(element)
            ]["knns_ind_member"]
        )  # original member data point always as first point
        knns_ind_non_member_current.append(
            loaded_obj["experiment {}".format(experiment)][
                "element {}".format(element)
            ]["knns_ind_non_member"]
        )
        knns_distances_member_current.append(
            loaded_obj["experiment {}".format(experiment)][
                "element {}".format(element)
            ]["knns_distances_member"]
        )
        knns_distances_non_member_current.append(
            loaded_obj["experiment {}".format(experiment)][
                "element {}".format(element)
            ]["knns_distances_non_member"]
        )
        # append current point values to list of all points in this experiment
        knns_ind_member_points.append(knns_ind_member_current)
        knns_ind_non_member_points.append(knns_ind_non_member_current)
        knns_distances_member_points.append(knns_distances_member_current)
        knns_distances_non_member_points.append(knns_distances_non_member_current)

    # append all points of experiment to list of all experiments
    knns_ind_member.append(knns_ind_member_points)
    knns_ind_non_member.append(knns_ind_non_member_points)
    knns_distances_member.append(knns_distances_member_points)
    knns_distances_non_member.append(knns_distances_non_member_points)
    knns_ind_member_points = []
    knns_ind_non_member_points = []
    knns_distances_member_points = []
    knns_distances_non_member_points = []

    train_accuracy.append(
        loaded_obj["experiment {}".format(experiment)]["element {}".format(element)][
            "train_accuracy"
        ][1]
    )
    test_accuracy.append(
        loaded_obj["experiment {}".format(experiment)]["element {}".format(element)][
            "test_accuracy"
        ][1]
    )
    scale.append(
        loaded_obj["experiment {}".format(experiment)]["element {}".format(element)][
            "scale"
        ]
    )
    k_neighbors.append(
        loaded_obj["experiment {}".format(experiment)]["element {}".format(element)][
            "k_neighbors"
        ]
    )
    amount_generate_neighbors.append(
        loaded_obj["experiment {}".format(experiment)]["element {}".format(element)][
            "amount_generate_neighbors"
        ]
    )

for i in range(len(loaded_obj["experiment 0"]["element 0"]["HYPERPARAMETERS"])):
    experiment_setups.append(
        "Experiment {}: {}".format(
            i, (loaded_obj["experiment 0"]["element 0"]["HYPERPARAMETERS"][i])
        )
    )

mean_knns_layers_member_all_experiments = []
mean_knns_layers_non_member_all_experiments = []
mean_distances_knns_layers_member_all_experiments = []
mean_distances_knns_layers_non_member_all_experiments = []
sum_similarities_knns_layers_member_all_experiments = []
sum_similarities_knns_layers_non_member_all_experiments = []
print("Got data from pickle.")
print("The experiment setups are: {}".format(experiment_setups))


# In[14]:



# prepare data for plots
for experiment in range(len(experiment_setups)):  # for all experiments
    for point in range(amount_m_nm_total):
        knns_indices_list_member = []
        knns_indices_list_non_member = []
        # make dicts to lists for get_differences_knns_btw_layers and to compare distances
        knns_indices_list_member = list(knns_ind_member[experiment][point][0].items())
        knns_indices_list_non_member = list(
            knns_ind_non_member[experiment][point][0].items()
        )
        # get for point: for layer: distance to k nns
        knns_distances_list_member = list(
            knns_distances_member[experiment][point][0].items()
        )
        knns_distances_list_non_member = list(
            knns_distances_non_member[experiment][point][0].items()
        )
        # outputs amount of changes in nn for one data point btw two layers (e.g. 'layer 1': [1])
        _, _, differences_knns_total_member = get_differences_knns_btw_layers(
            1,
            knns_indices_list_member,
            compares_with_first_layer_only=False,
        )  # 1 because in each fprop of DkNN 1 input point != amount_m_nm_total (for how many this is repeated)
        _, _, differences_knns_total_non_member = get_differences_knns_btw_layers(
            1,
            knns_indices_list_non_member,
            compares_with_first_layer_only=False,
        )  # 1 because in each fprop of DkNN 1 input point != amount_m_nm_total (for how many this is repeated)
        # similarities_knns_total_member e.g. {layer 1: [100], layer 2: [100],...}
        _, _, similarities_knns_total_member = get_similarities_knns_btw_layers(
            1,
            knns_indices_list_member,
            compares_with_first_layer_only=True,
        )
        _, _, similarities_knns_total_non_member = get_similarities_knns_btw_layers(
            1,
            knns_indices_list_non_member,
            compares_with_first_layer_only=True,
        )
        distances_knns_all_member = get_distances_of_knns(1, knns_distances_list_member)
        distances_knns_all_non_member = get_distances_of_knns(
            1, knns_distances_list_non_member
        )
        # put amount of changes (differences_knns_total_member)/ distances knns for all points of one experiments in one dict
        if point == 0:
            differences_knns_total_member_points_experiment = (
                differences_knns_total_member
            )
            differences_knns_total_non_member_points_experiment = (
                differences_knns_total_non_member
            )

            similarities_knns_total_member_points_experiment = (
                similarities_knns_total_member
            )
            similarities_knns_total_non_member_points_experiment = (
                similarities_knns_total_non_member
            )

            distances_knns_total_member_points_experiment = distances_knns_all_member
            distances_knns_total_non_member_points_experiment = (
                distances_knns_all_non_member
            )

        else:
            # changes are always between two layers so range starts at 1 (means changes btw. layer 0 and 1)
            for layer in range(1, len(knns_indices_list_member)):
                differences_knns_total_member_points_experiment[
                    "layer {}".format(layer)
                ].append(differences_knns_total_member["layer {}".format(layer)][0])
                differences_knns_total_non_member_points_experiment[
                    "layer {}".format(layer)
                ].append(differences_knns_total_non_member["layer {}".format(layer)][0])

                similarities_knns_total_member_points_experiment[
                    "layer {}".format(layer)
                ].append(similarities_knns_total_member["layer {}".format(layer)][0])
                similarities_knns_total_non_member_points_experiment[
                    "layer {}".format(layer)
                ].append(
                    similarities_knns_total_non_member["layer {}".format(layer)][0]
                )
            # distances of knn are measured for each layer so range starts at 0
            for layer in range(0, len(knns_indices_list_member)):
                distances_knns_total_member_points_experiment[
                    "layer {}".format(layer)
                ].append(distances_knns_all_member["layer {}".format(layer)][0])
                distances_knns_total_non_member_points_experiment[
                    "layer {}".format(layer)
                ].append(distances_knns_all_non_member["layer {}".format(layer)][0])
    # outputs mean of changes in nn/ distances for one experiment
    mean_knns_layers_member_one_experiment = get_mean_knns_layer(
        knns_indices_list_member, differences_knns_total_member_points_experiment
    )
    mean_knns_layers_non_member_one_experiment = get_mean_knns_layer(
        knns_indices_list_non_member,
        differences_knns_total_non_member_points_experiment,
    )

    mean_distances_knns_layers_member_one_experiment = get_mean_distances_of_knns(
        distances_knns_total_member_points_experiment, knns_distances_list_member
    )
    mean_distances_knns_layers_non_member_one_experiment = get_mean_distances_of_knns(
        distances_knns_total_non_member_points_experiment,
        knns_distances_list_non_member,
    )
    # get sum of similarities for one experiment for all points (e.g. [1,1,2,1] + [1,1,0,1] --> [2,2,2,2])
    sum_similarities_knns_layers_member_one_experiment = get_sum_similarities_of_knns(
        similarities_knns_total_member_points_experiment, knns_indices_list_member
    )
    sum_similarities_knns_layers_non_member_one_experiment = (
        get_sum_similarities_of_knns(
            similarities_knns_total_non_member_points_experiment,
            knns_indices_list_non_member,
        )
    )
    # append means for one experiment to all experiments
    mean_knns_layers_member_all_experiments.append(
        mean_knns_layers_member_one_experiment
    )
    mean_knns_layers_non_member_all_experiments.append(
        mean_knns_layers_non_member_one_experiment
    )
    # get for all experiments per experiment the mean distance per layer (5 in total)
    mean_distances_knns_layers_member_all_experiments.append(
        mean_distances_knns_layers_member_one_experiment
    )
    mean_distances_knns_layers_non_member_all_experiments.append(
        mean_distances_knns_layers_non_member_one_experiment
    )
    # get for all experiments the sum of similarities in one list (e.g. [[2,2,2,2][1,2,3,4]]
    sum_similarities_knns_layers_member_all_experiments.append(
        sum_similarities_knns_layers_member_one_experiment
    )
    sum_similarities_knns_layers_non_member_all_experiments.append(
        sum_similarities_knns_layers_non_member_one_experiment
    )
# get the total sum for all layers for one experiment (e.g. [2,2,2,2] --> [8])
sum_similarities_knns_member_all_experiment = []
sum_similarities_knns_non_member_all_experiment = []
for experiment in range(len(sum_similarities_knns_layers_member_all_experiments)):
    sum_experiment_member = np.sum(
        sum_similarities_knns_layers_member_all_experiments[experiment]
    )
    sum_experiment_non_member = np.sum(
        sum_similarities_knns_layers_non_member_all_experiments[experiment]
    )
    sum_similarities_knns_member_all_experiment.append(sum_experiment_member)
    sum_similarities_knns_non_member_all_experiment.append(sum_experiment_non_member)
print("Data is ready to be plotted.")


# In[15]:


print("Plot mean of knns changes between two layers.")
print("In other words: \n")
print("1. Compare the knns of two consecutive layers --> save changes in knns (e.g. layer 0 [1,2], layer 1 [2,3] --> one change) --> repeat for all layers, all points\n")
print("2. Compute mean of changes btw. two consecutive layers for all member/ non-member points for one experiment.")
plot_member_non_member_layer_experiments_scatter(mean_knns_layers_member_all_experiments, mean_knns_layers_non_member_all_experiments, experiment_setups, 0, "/home/inafen/jupyter_notebooks/changes_knns_0.png", train_accuracy, test_accuracy)
plot_member_non_member_layer_experiments_scatter(mean_knns_layers_member_all_experiments, mean_knns_layers_non_member_all_experiments, experiment_setups, 1, "/home/inafen/jupyter_notebooks/changes_knns_1.png", train_accuracy, test_accuracy)
plot_member_non_member_layer_experiments_scatter(mean_knns_layers_member_all_experiments, mean_knns_layers_non_member_all_experiments, experiment_setups, 2, "/home/inafen/jupyter_notebooks/changes_knns_2.png", train_accuracy, test_accuracy)
plot_member_non_member_layer_experiments_scatter(mean_knns_layers_member_all_experiments, mean_knns_layers_non_member_all_experiments, experiment_setups, 3, "/home/inafen/jupyter_notebooks/changes_knns_3.png", train_accuracy, test_accuracy)


# In[16]:


print("Same as above (mean of knns changes between two layers), only in one plot to have a better overview")

plot_mean_layer_experiments_all(mean_knns_layers_member_all_experiments, mean_knns_layers_non_member_all_experiments, experiment_setups, 0, "/home/inafen/jupyter_notebooks/changes_knns_all.png", train_accuracy, test_accuracy)


# In[17]:


print("Plot mean of knns distances for each layer.")
print("In other words: \n")
print("1. Get distances to point of all knns for one layer --> --> repeat for all layers, all points\n")
print("2. Compute mean of knn distances in one layer for all member/ non-member points for one experiment.")



plot_member_non_member_layer_experiments_scatter(mean_distances_knns_layers_member_all_experiments, mean_distances_knns_layers_non_member_all_experiments, experiment_setups, 0, "/home/inafen/jupyter_notebooks/distances_knns_0.png", train_accuracy, test_accuracy, ylabel= "Mean distance of knns", suptitle = "Mean distance of knns in layer {}")
plot_member_non_member_layer_experiments_scatter(mean_distances_knns_layers_member_all_experiments, mean_distances_knns_layers_non_member_all_experiments, experiment_setups, 1, "/home/inafen/jupyter_notebooks/distances_knns_1.png", train_accuracy, test_accuracy, ylabel= "Mean distance of knns", suptitle = "Mean distance of knns in layer {}")
plot_member_non_member_layer_experiments_scatter(mean_distances_knns_layers_member_all_experiments, mean_distances_knns_layers_non_member_all_experiments, experiment_setups, 2, "/home/inafen/jupyter_notebooks/distances_knns_2.png", train_accuracy, test_accuracy, ylabel= "Mean distance of knns", suptitle = "Mean distance of knns in layer {}")
plot_member_non_member_layer_experiments_scatter(mean_distances_knns_layers_member_all_experiments, mean_distances_knns_layers_non_member_all_experiments, experiment_setups, 3, "/home/inafen/jupyter_notebooks/distances_knns_3.png", train_accuracy, test_accuracy, ylabel= "Mean distance of knns", suptitle = "Mean distance of knns in layer {}")
plot_member_non_member_layer_experiments_scatter(mean_distances_knns_layers_member_all_experiments, mean_distances_knns_layers_non_member_all_experiments, experiment_setups, 4, "/home/inafen/jupyter_notebooks/distances_knns_4.png", train_accuracy, test_accuracy, ylabel= "Mean distance of knns", suptitle = "Mean distance of knns in layer {}")


# ## *ADDED*: Analysis of whether mean distances are really 0.00
# 
# The graph inaccurately shows small distances as 0. But a look inside the data showed that the distances are in reality just very small. In most of the experiments, the mean distance of member knns is smaller than of non-member knns (see below).
# 
# It might be helpful to do a similar analysis for the other graphs down below in the future.

# In[23]:


print("Since plots are too unspecific to show minor differences (e.g. shows as 0.00 instead of real value 0.002), let's compare the values directly")
layer_0_member_smaller_distance = []
layer_1_member_smaller_distance = []
layer_2_member_smaller_distance = []
layer_3_member_smaller_distance = []
layer_4_member_smaller_distance = []
for experiment in range(len(mean_distances_knns_layers_member_all_experiments)):
    #checks: Is in this experiment the mean distance of the knns for layer x of members smaller than of non-members? If yes, append True, else False
    layer_0_member_smaller_distance.append(mean_distances_knns_layers_member_all_experiments[experiment][0]<mean_distances_knns_layers_non_member_all_experiments[experiment][0])
    layer_1_member_smaller_distance.append(mean_distances_knns_layers_member_all_experiments[experiment][1]<mean_distances_knns_layers_non_member_all_experiments[experiment][1])
    layer_2_member_smaller_distance.append(mean_distances_knns_layers_member_all_experiments[experiment][2]<mean_distances_knns_layers_non_member_all_experiments[experiment][2])
    layer_3_member_smaller_distance.append(mean_distances_knns_layers_member_all_experiments[experiment][3]<mean_distances_knns_layers_non_member_all_experiments[experiment][3])
    layer_4_member_smaller_distance.append(mean_distances_knns_layers_member_all_experiments[experiment][4]<mean_distances_knns_layers_non_member_all_experiments[experiment][4])
print("The mean distance of knns in layer 0 of member data is in {} of {} experiments smaller than the mean distance of non-member data.".format(layer_0_member_smaller_distance.count(True), len(layer_0_member_smaller_distance)))
print("The mean distance of knns in layer 1 of member data is in {} of {} experiments smaller than the mean distance of non-member data.".format(layer_1_member_smaller_distance.count(True), len(layer_0_member_smaller_distance)))
print("The mean distance of knns in layer 2 of member data is in {} of {} experiments smaller than the mean distance of non-member data.".format(layer_2_member_smaller_distance.count(True), len(layer_0_member_smaller_distance)))
print("The mean distance of knns in layer 3 of member data is in {} of {} experiments smaller than the mean distance of non-member data.".format(layer_3_member_smaller_distance.count(True), len(layer_0_member_smaller_distance)))
print("The mean distance of knns in layer 4 of member data is in {} of {} experiments smaller than the mean distance of non-member data.".format(layer_4_member_smaller_distance.count(True), len(layer_0_member_smaller_distance)))


# In[18]:


print("Plot sum of consistent knns (= nns, that stay knns) btw. layers in DkNN")
print("In other words: \n")
print("1. Compare the knns of the first and another layer --> save consistent knns (e.g. layer 0 [1,2], layer 1 [2,3] --> one consistent) --> repeat for all layers, all points\n")
print("2. Compute sum of consistent knns throughout the whole DkNN for all member/ non-member points for one experiment.")


plot_member_non_member_layer_experiments_scatter(sum_similarities_knns_layers_member_all_experiments, sum_similarities_knns_layers_non_member_all_experiments, experiment_setups, 0, "/home/inafen/jupyter_notebooks/similarities_knns_0.png", train_accuracy, test_accuracy, ylabel= "Sum of consistent knns", suptitle = "Sum of consistent knns (= nns, that stay knns) btw. layer 0 & 1")
plot_member_non_member_layer_experiments_scatter(sum_similarities_knns_layers_member_all_experiments, sum_similarities_knns_layers_non_member_all_experiments, experiment_setups, 1, "/home/inafen/jupyter_notebooks/similarities_knns_1.png", train_accuracy, test_accuracy, ylabel= "Sum of consistent knns", suptitle = "Sum of consistent knns (= nns, that stay knns) btw. layer 0 & 2")
plot_member_non_member_layer_experiments_scatter(sum_similarities_knns_layers_member_all_experiments, sum_similarities_knns_layers_non_member_all_experiments, experiment_setups, 2, "/home/inafen/jupyter_notebooks/similarities_knns_2.png", train_accuracy, test_accuracy, ylabel= "Sum of consistent knns", suptitle = "Sum of consistent knns (= nns, that stay knns) btw. layer 0 & 3")
plot_member_non_member_layer_experiments_scatter(sum_similarities_knns_layers_member_all_experiments, sum_similarities_knns_layers_non_member_all_experiments, experiment_setups, 3, "/home/inafen/jupyter_notebooks/similarities_knns_3.png", train_accuracy, test_accuracy, ylabel= "Sum of consistent knns", suptitle = "Sum of consistent knns (= nns, that stay knns) btw. layer 0 & 4")


# In[19]:


print("Plot sum of consistent knns (= nns, that stay knns) in whole DkNN")
print("Almost same as above, but with one difference: \n")
print("As a last step, the sum of all consistent knns for all layers for all members/non-members is calculated, for each experiment.")

plot_member_non_member_experiments_histogram(sum_similarities_knns_member_all_experiment, sum_similarities_knns_non_member_all_experiment, train_accuracy, test_accuracy, [0,5000,10000,15000,20000], "/home/inafen/jupyter_notebooks/similarities_knns_big.png")
plot_member_non_member_experiments_histogram(sum_similarities_knns_member_all_experiment, sum_similarities_knns_non_member_all_experiment, train_accuracy, test_accuracy, [0,2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000], "/home/inafen/jupyter_notebooks/similarities_knns_small.png")


# In[ ]:




