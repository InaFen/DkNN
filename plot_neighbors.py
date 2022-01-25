import pickle
from utils.utils import *
from utils.utils_plot import (
    plot_member_non_member_layer_experiments_scatter,
    plot_mean_layer_experiments_all,
    plot_member_non_member_experiments_histogram,
)

# open pickle
with open(
    "/home/inafen/jupyter_notebooks/data_neighbors_changed_non_member.pickle", "rb"
) as f:
    loaded_obj = pickle.load(f)

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


plot_member_non_member_layer_experiments_scatter(
    mean_knns_layers_member_all_experiments,
    mean_knns_layers_non_member_all_experiments,
    experiment_setups,
    0,
    "/home/inafen/jupyter_notebooks/changes_knns_0.png",
    train_accuracy,
    test_accuracy,
)
plot_member_non_member_layer_experiments_scatter(
    mean_knns_layers_member_all_experiments,
    mean_knns_layers_non_member_all_experiments,
    experiment_setups,
    1,
    "/home/inafen/jupyter_notebooks/changes_knns_1.png",
    train_accuracy,
    test_accuracy,
)
plot_member_non_member_layer_experiments_scatter(
    mean_knns_layers_member_all_experiments,
    mean_knns_layers_non_member_all_experiments,
    experiment_setups,
    2,
    "/home/inafen/jupyter_notebooks/changes_knns_2.png",
    train_accuracy,
    test_accuracy,
)
plot_member_non_member_layer_experiments_scatter(
    mean_knns_layers_member_all_experiments,
    mean_knns_layers_non_member_all_experiments,
    experiment_setups,
    3,
    "/home/inafen/jupyter_notebooks/changes_knns_3.png",
    train_accuracy,
    test_accuracy,
)
plot_mean_layer_experiments_all(
    mean_knns_layers_member_all_experiments,
    mean_knns_layers_non_member_all_experiments,
    experiment_setups,
    0,
    "/home/inafen/jupyter_notebooks/changes_knns_all.png",
    train_accuracy,
    test_accuracy,
)

plot_member_non_member_layer_experiments_scatter(
    mean_distances_knns_layers_member_all_experiments,
    mean_distances_knns_layers_non_member_all_experiments,
    experiment_setups,
    0,
    "/home/inafen/jupyter_notebooks/distances_knns_0.png",
    train_accuracy,
    test_accuracy,
    ylabel="Mean distance of knns",
    suptitle="Mean distance of knns in layer {}",
)
plot_member_non_member_layer_experiments_scatter(
    mean_distances_knns_layers_member_all_experiments,
    mean_distances_knns_layers_non_member_all_experiments,
    experiment_setups,
    1,
    "/home/inafen/jupyter_notebooks/distances_knns_1.png",
    train_accuracy,
    test_accuracy,
    ylabel="Mean distance of knns",
    suptitle="Mean distance of knns in layer {}",
)
plot_member_non_member_layer_experiments_scatter(
    mean_distances_knns_layers_member_all_experiments,
    mean_distances_knns_layers_non_member_all_experiments,
    experiment_setups,
    2,
    "/home/inafen/jupyter_notebooks/distances_knns_2.png",
    train_accuracy,
    test_accuracy,
    ylabel="Mean distance of knns",
    suptitle="Mean distance of knns in layer {}",
)
plot_member_non_member_layer_experiments_scatter(
    mean_distances_knns_layers_member_all_experiments,
    mean_distances_knns_layers_non_member_all_experiments,
    experiment_setups,
    3,
    "/home/inafen/jupyter_notebooks/distances_knns_3.png",
    train_accuracy,
    test_accuracy,
    ylabel="Mean distance of knns",
    suptitle="Mean distance of knns in layer {}",
)
plot_member_non_member_layer_experiments_scatter(
    mean_distances_knns_layers_member_all_experiments,
    mean_distances_knns_layers_non_member_all_experiments,
    experiment_setups,
    4,
    "/home/inafen/jupyter_notebooks/distances_knns_4.png",
    train_accuracy,
    test_accuracy,
    ylabel="Mean distance of knns",
    suptitle="Mean distance of knns in layer {}",
)

plot_member_non_member_layer_experiments_scatter(
    sum_similarities_knns_layers_member_all_experiments,
    sum_similarities_knns_layers_non_member_all_experiments,
    experiment_setups,
    0,
    "/home/inafen/jupyter_notebooks/similarities_knns_0.png",
    train_accuracy,
    test_accuracy,
    ylabel="Sum of consistent knns",
    suptitle="Sum of consistent knns (= nns, that stay knns) btw. layer {} & {}",
)
plot_member_non_member_layer_experiments_scatter(
    sum_similarities_knns_layers_member_all_experiments,
    sum_similarities_knns_layers_non_member_all_experiments,
    experiment_setups,
    1,
    "/home/inafen/jupyter_notebooks/similarities_knns_1.png",
    train_accuracy,
    test_accuracy,
    ylabel="Sum of consistent knns",
    suptitle="Sum of consistent knns (= nns, that stay knns) btw. layer {} & {}",
)
plot_member_non_member_layer_experiments_scatter(
    sum_similarities_knns_layers_member_all_experiments,
    sum_similarities_knns_layers_non_member_all_experiments,
    experiment_setups,
    2,
    "/home/inafen/jupyter_notebooks/similarities_knns_2.png",
    train_accuracy,
    test_accuracy,
    ylabel="Sum of consistent knns",
    suptitle="Sum of consistent knns (= nns, that stay knns) btw. layer {} & {}",
)
plot_member_non_member_layer_experiments_scatter(
    sum_similarities_knns_layers_member_all_experiments,
    sum_similarities_knns_layers_non_member_all_experiments,
    experiment_setups,
    3,
    "/home/inafen/jupyter_notebooks/similarities_knns_3.png",
    train_accuracy,
    test_accuracy,
    ylabel="Sum of consistent knns",
    suptitle="Sum of consistent knns (= nns, that stay knns) btw. layer {} & {}",
)

plot_member_non_member_experiments_histogram(
    sum_similarities_knns_member_all_experiment,
    sum_similarities_knns_non_member_all_experiment,
    train_accuracy,
    test_accuracy,
    [0, 5000, 10000, 15000, 20000],
    "/home/inafen/jupyter_notebooks/similarities_knns_big.png",
)
plot_member_non_member_experiments_histogram(
    sum_similarities_knns_member_all_experiment,
    sum_similarities_knns_non_member_all_experiment,
    train_accuracy,
    test_accuracy,
    [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
    "/home/inafen/jupyter_notebooks/similarities_knns_small.png",
)
