import pickle
from utils import *
import matplotlib.pyplot as plt


with open('/home/inafen/jupyter_notebooks/data_neighbors_test1.pickle', 'rb') as f:
    loaded_obj = pickle.load(f)

train_accuracy = []
test_accuracy = []
knns_ind_member = []
knns_ind_non_member = []
knns_distances_member =[]
knns_distances_non_member = []
scale = []
k_neighbors = []
amount_generate_neighbors = []
experiment_setups = []

amount_m_nm_total = loaded_obj[0][0][10]
for experiment in range(len(loaded_obj)):
    for point in range(amount_m_nm_total):
        knns_ind_member_current = []
        knns_ind_non_member_current = []
        knns_distances_member_current = []
        knns_distances_non_member_current = []

        knns_ind_member_current.append(loaded_obj[experiment][point][2]) #original member data point always as first point
        knns_ind_non_member_current.append(loaded_obj[experiment][point][3])
        knns_distances_member_current.append(loaded_obj[experiment][point][4])
        knns_distances_non_member_current.append(loaded_obj[experiment][point][5])
    knns_ind_member.append(knns_ind_member_current)
    knns_ind_non_member.append(knns_ind_non_member_current)
    knns_distances_member.append(knns_distances_member_current)
    knns_distances_non_member.append(knns_distances_non_member_current)

    train_accuracy.append(loaded_obj[experiment][point][0][1])
    test_accuracy.append(loaded_obj[experiment][point][1][1])
    scale.append(loaded_obj[experiment][point][6])
    k_neighbors.append(loaded_obj[experiment][point][7])
    amount_generate_neighbors.append(loaded_obj[experiment][point][8])
    experiment_setups.append("Experiment {}: {}".format(experiment,(loaded_obj[experiment][point][9])))

mean_knns_layers_member_all = []
mean_knns_layers_non_member_all = []

#for all experiments
for experiment in range(len(experiment_setups)):
    knns_indices_list_member = []
    knns_indices_list_non_member = []
    knns_indices_list_member = list(knns_ind_member[experiment][0].items())
    knns_indices_list_non_member = list(knns_ind_non_member[experiment][0].items())
    #outputs amount of changes in nn for each data point btw two layers (e.g. 'layer 1': [1, 1])
    _, _, differences_knns_total_member = get_differences_knns_btw_layers(amount_m_nm_total, knns_indices_list_member, num_instead_of_data=True)
    _, _, differences_knns_total_non_member = get_differences_knns_btw_layers(amount_m_nm_total, knns_indices_list_non_member, num_instead_of_data=True)
    #outputs mean of changes in nn
    mean_knns_layers_member = get_mean_knns_layer(knns_indices_list_member, differences_knns_total_member)
    mean_knns_layers_non_member = get_mean_knns_layer(knns_indices_list_non_member, differences_knns_total_non_member)
    mean_knns_layers_member_all.append(mean_knns_layers_member)
    mean_knns_layers_non_member_all.append(mean_knns_layers_non_member)


def plot_mean_layer_experiments(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups, layer, filepath , train_accuracy, test_accuracy):
    """
    Plot the mean of changes in knns btw to layers.

    :param mean_knns_layers_member_all: changes of knns btw two layers for member data
    :param mean_knns_layers_non_member_all: changes of knns btw two layers for non member data
    :param experiment_setups: setups of all experiments
    :param layer: which layer (or more specific: between two layers) is looked at
    :param filepath: where graph should be saved
    :param train_accuracy: accuracy of training
    :param test_accuracy: accuracy of testing
    """
    x_member = []
    y_member_1 = []
    #y_member_2 = []
    #y_member_3 = []
    #y_member_4 = []

    x_non_member = []
    y_non_member_1 = []
    #y_non_member_2 = []
    #y_non_member_3 = []
    #y_non_member_4 = []
    # TODO subplots with two datasets (# lines)

    for experiment in range(len(experiment_setups)):
        x_member.append(experiment)
        x_non_member.append(experiment)
        y_member_1.append(mean_knns_layers_member_all[experiment][layer])
        y_non_member_1.append(mean_knns_layers_non_member_all[experiment][layer])
        #y_member_2.append(mean_knns_layers_member_all[experiment][layer+1])
        #y_non_member_2.append(mean_knns_layers_non_member_all[experiment][layer+1])
        #y_member_3.append(mean_knns_layers_member_all[experiment][layer]+2)
        #y_non_member_3.append(mean_knns_layers_non_member_all[experiment][layer+2])
        #y_member_4.append(mean_knns_layers_member_all[experiment][layer+3])
        #y_non_member_4.append(mean_knns_layers_non_member_all[experiment][layer+3])
       # xticks.append(experiment_setups[experiment])

    plt.scatter(x_member, y_member_1, label = "Member")
    plt.scatter(x_non_member, y_non_member_1, label = "Non-Member")
    plt.plot(x_member, y_member_1, '-o')
    plt.plot(x_non_member, y_non_member_1, '-o')


    X_axis = np.arange(len(experiment_setups))
    #TODO change xticks to experiment_setups
    plt.xticks(X_axis, ["exp 1", "exp 2","exp 3","exp 4","exp 5","exp 6"])
    plt.xlabel("Experiments")
    plt.ylabel("Mean of knn changes")
    plt.suptitle("Mean of changes in knns btw. layer {} & {}".format(layer, layer+1), fontsize = 18)
    plt.title("Model: Lenet-5, Train Acc: {}, Test Acc: {}".format(round(train_accuracy[1], ndigits=4), round(test_accuracy[1], ndigits=4)), fontsize = 10)
    plt.legend()
    plt.show()
    plt.savefig(filepath, bbox_inches="tight")
    plt.clf()

plot_mean_layer_experiments(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups, 0, "/home/inafen/jupyter_notebooks/changes_knns_0.png", train_accuracy, test_accuracy)
plot_mean_layer_experiments(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups, 1, "/home/inafen/jupyter_notebooks/changes_knns_1.png", train_accuracy, test_accuracy)
plot_mean_layer_experiments(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups, 2, "/home/inafen/jupyter_notebooks/changes_knns_2.png", train_accuracy, test_accuracy)
plot_mean_layer_experiments(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups, 3, "/home/inafen/jupyter_notebooks/changes_knns_3.png", train_accuracy, test_accuracy)