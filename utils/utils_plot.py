import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange
import seaborn as sns
import pandas as pd


def plot_reliability_diagram(confidence, labels, filepath):
    """
    Takes in confidence values for predictions and correct
    labels for the data, plots a reliability diagram.

    :param confidence: nb_samples x nb_classes (e.g., output of softmax)
    :param labels: vector of nb_samples
    :param filepath: where to save the diagram
    """
    print(confidence.shape[0])
    print(labels.shape[0])
    assert len(confidence.shape) == 2
    assert len(labels.shape) == 1
    assert confidence.shape[0] == labels.shape[0]
    print("Saving reliability diagram at: " + str(filepath))
    if confidence.max() <= 1.0:
        # confidence array is output of softmax
        bins_start = [b / 10.0 for b in xrange(0, 10)]
        bins_end = [b / 10.0 for b in xrange(1, 11)]
        bins_center = [(b + 0.5) / 10.0 for b in xrange(0, 10)]
        preds_conf = np.max(confidence, axis=1)
        preds_l = np.argmax(confidence, axis=1)
    else:
        raise ValueError("Confidence values go above 1.")

    print(preds_conf.shape, preds_l.shape)

    # Create var for reliability diagram
    # Will contain mean accuracies for each bin
    reliability_diag = []
    num_points = []  # keeps the number of points in each bar

    # Find average accuracy per confidence bin
    for bin_start, bin_end in zip(bins_start, bins_end):
        above = preds_conf >= bin_start
        if bin_end == 1.0:
            below = preds_conf <= bin_end
        else:
            below = preds_conf < bin_end
        mask = np.multiply(above, below)
        num_points.append(np.sum(mask))
        bin_mean_acc = max(0, np.mean(preds_l[mask] == labels[mask]))
        reliability_diag.append(bin_mean_acc)

    # Plot diagram
    assert len(reliability_diag) == len(bins_center)
    print(reliability_diag)
    print(bins_center)
    print(num_points)
    fig, ax1 = plt.subplots()
    _ = ax1.bar(bins_center, reliability_diag, width=0.1, alpha=0.8)
    plt.xlim([0, 1.0])
    ax1.set_ylim([0, 1.0])

    ax2 = ax1.twinx()
    print(sum(num_points))
    ax2.plot(bins_center, num_points, color="r", linestyle="-", linewidth=7.0)
    ax2.set_ylabel("Number of points in the data", fontsize=16, color="r")

    if len(np.argwhere(confidence[0] != 0.0)) == 1:
        # This is a DkNN diagram
        ax1.set_xlabel("Prediction Credibility", fontsize=16)
    else:
        # This is a softmax diagram
        ax1.set_xlabel("Prediction Confidence", fontsize=16)
    ax1.set_ylabel("Prediction Accuracy", fontsize=16)
    ax1.tick_params(axis="both", labelsize=14)
    ax2.tick_params(axis="both", labelsize=14, colors="r")
    fig.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()


def plot_mean_knns_layer(
    knns_attribute, filepath, differences_knns_total, ylabel, xlabel, title
):
    """
    Creates bar graph of mean of knns for each layer for one data set.

    :param knns_attribute: indices or labels of knns
    :param filepath: filepath to where plot should be saved.
    :param differences_knns_total: amount of knns changes for data per layer
    """

    # get mean of knns per layer
    mean_knns_layers = []
    for layer in range(1, len(knns_attribute)):
        mean_knns_layers.append(
            np.mean(differences_knns_total.get("layer {}".format(layer)))
        )

    # plot mean

    layers = []
    for layer in range(1, len(knns_attribute)):
        layers.append("layer {}".format(layer))
    x_pos = np.arange(len(layers))
    plt.bar(x_pos, mean_knns_layers)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_pos, layers)
    plt.show()

    plt.savefig(filepath, bbox_inches="tight")


def plot_changes_knns_3(
    mean_layers_1,
    mean_layers_2,
    mean_layers_3,
    knns_attribute,
    filepath,
    xlabel,
    ylabel,
    title,
    label_1="Train",
    label_2="Test",
    label_3="Noisy",
):
    """
    Plot changes/differences of an knns attribute for 3 data sets (train, test, noisy).

    :param mean_layers_1: mean of attribute per layer/btw. two layers for data set
    :param mean_layers_2: mean of attribute per layer/btw. two layers for data set
    :param mean_layers_3: mean of attribute per layer/btw. two layers for data set
    :param knns_attribute: indices/labels/distances of knns
    :param filepath: filepath
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param title: title
    :param label_1: label for bar plot 1
    :param label_2: label for bar plot 2
    :param label_3: label for bar plot 3
    :return:
    """
    layers = []
    for layer in range(1, len(knns_attribute)):
        layers.append("layer {}".format(layer))

    X_axis = np.arange(len(layers))

    plt.bar(X_axis - 0.2, mean_layers_1, 0.2, label=label_1)
    plt.bar(X_axis, mean_layers_2, 0.2, label=label_2)
    plt.bar(X_axis + 0.2, mean_layers_3, 0.2, label=label_3)

    plt.xticks(X_axis, layers)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(filepath, bbox_inches="tight")
    plt.clf()

def plot_member_non_member_layer_experiments_scatter(attribute_knns_layers_member_all, attribute_knns_layers_non_member_all, experiment_setups, layer, filepath , train_accuracy, test_accuracy, ylabel = "Mean of knn changes", suptitle = "Mean of changes in knns btw. layer {} & {}" ):
    """
    Plot an attribute (e.g. the mean of changes in knns btw two layers) of the member and non-member data of all experiments for one layer/ for inbetween two layers as a scatter plot.

    :param attribute_knns_layers_member_all: Attribute of knn of member data, e.g. changes of knns btw two layers for member data or distances of knns, for all layers
    :param attribute_knns_layers_non_member_all: Attribute of knn of non member data, e.g. changes of knns btw two layers for non member data or distances of knns, for all layers
    :param experiment_setups: setups of all experiments
    :param layer: which layer (or depending on attribute: between two layers) is looked at
    :param filepath: where graph should be saved
    :param train_accuracy: accuracy of training
    :param test_accuracy: accuracy of testing
    """
    x_member = []
    y_member_1 = []

    x_non_member = []
    y_non_member_1 = []

    for experiment in range(len(experiment_setups)):
        x_member.append(experiment)
        x_non_member.append(experiment)
        y_member_1.append(attribute_knns_layers_member_all[experiment][layer])
        y_non_member_1.append(attribute_knns_layers_non_member_all[experiment][layer])

    plt.scatter(x_member, y_member_1, label = "Member")
    plt.scatter(x_non_member, y_non_member_1, label = "Non-Member")
    plt.plot(x_member, y_member_1, linestyle = 'None') #'-o' to connect dots if wanted
    plt.plot(x_non_member, y_non_member_1, linestyle = 'None') #'-o' to connect dots if wanted


    X_axis = np.arange(len(experiment_setups))
    plt.xticks(X_axis, experiment_setups, fontsize = 5, rotation = 90)
    plt.xlabel("Experiments")
    plt.ylabel(ylabel)
    plt.suptitle(suptitle.format(layer, layer+1), fontsize = 18)
    plt.title("Model: Lenet-5, Train Acc: {}, Test Acc: {}".format(round(train_accuracy[1], ndigits=4), round(test_accuracy[1], ndigits=4)), fontsize = 10)
    plt.legend()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.clf()


def plot_mean_layer_experiments_all(mean_knns_layers_member_all, mean_knns_layers_non_member_all, experiment_setups,
                                    layer, filepath, train_accuracy, test_accuracy):
    """
    To plot the mean of changes in knns btw. two layers, for all layers.
    This function is specialised for only this use and has to be generalised to be able to be used more broadly. #TODO

    :param mean_knns_layers_member_all:
    :param mean_knns_layers_non_member_all:
    :param experiment_setups:
    :param layer:
    :param filepath:
    :param train_accuracy:
    :param test_accuracy:
    :return:
    """

    x_member = []
    y_member_1 = []
    y_member_2 = []
    y_member_3 = []
    y_member_4 = []

    x_non_member = []
    y_non_member_1 = []
    y_non_member_2 = []
    y_non_member_3 = []
    y_non_member_4 = []

    for experiment in range(len(experiment_setups)):
        x_member.append(experiment)
        x_non_member.append(experiment)
        y_member_1.append(mean_knns_layers_member_all[experiment][layer])
        y_non_member_1.append(mean_knns_layers_non_member_all[experiment][layer])
        y_member_2.append(mean_knns_layers_member_all[experiment][layer + 1])
        y_non_member_2.append(mean_knns_layers_non_member_all[experiment][layer + 1])
        y_member_3.append(mean_knns_layers_member_all[experiment][layer] + 2)
        y_non_member_3.append(mean_knns_layers_non_member_all[experiment][layer + 2])
        y_member_4.append(mean_knns_layers_member_all[experiment][layer + 3])
        y_non_member_4.append(mean_knns_layers_non_member_all[experiment][layer + 3])
    # xticks.append(experiment_setups[experiment])

    member_dict_1 = {'Experiments': x_member, 'Mean of knn changes': y_member_1}
    non_member_dict_1 = {'Experiments': x_non_member, 'Mean of knn changes': y_non_member_1}
    member_dataframe_1 = pd.DataFrame(member_dict_1)
    non_member_dataframe_1 = pd.DataFrame(non_member_dict_1)
    concatenated_1 = pd.concat(
        [member_dataframe_1.assign(dataset='member'), non_member_dataframe_1.assign(dataset='non_member')])

    member_dict_2 = {'Experiments': x_member, 'Mean of knn changes': y_member_2}
    non_member_dict_2 = {'Experiments': x_non_member, 'Mean of knn changes': y_non_member_2}
    member_dataframe_2 = pd.DataFrame(member_dict_2)
    non_member_dataframe_2 = pd.DataFrame(non_member_dict_2)
    concatenated_2 = pd.concat(
        [member_dataframe_2.assign(dataset='member2'), non_member_dataframe_2.assign(dataset='non_member2')])

    member_dict_3 = {'Experiments': x_member, 'Mean of knn changes': y_member_3}
    non_member_dict_3 = {'Experiments': x_non_member, 'Mean of knn changes': y_non_member_3}
    member_dataframe_3 = pd.DataFrame(member_dict_3)
    non_member_dataframe_3 = pd.DataFrame(non_member_dict_3)
    concatenated_3 = pd.concat(
        [member_dataframe_3.assign(dataset='member3'), non_member_dataframe_3.assign(dataset='non_member3')])

    member_dict_4 = {'Experiments': x_member, 'Mean of knn changes': y_member_4}
    non_member_dict_4 = {'Experiments': x_non_member, 'Mean of knn changes': y_non_member_4}
    member_dataframe_4 = pd.DataFrame(member_dict_4)
    non_member_dataframe_4 = pd.DataFrame(non_member_dict_4)
    concatenated_4 = pd.concat(
        [member_dataframe_4.assign(dataset='member4'), non_member_dataframe_4.assign(dataset='non_member4')])

    sns.set()
    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.suptitle('Mean of changes in knns btw. layers (Model: Lenet-5, Train Acc: {}, Test Acc: {})'.format(
        round(train_accuracy[1], ndigits=4), round(test_accuracy[1], ndigits=4)))
    sns.scatterplot(ax=axes[0], x='Experiments', y='Mean of knn changes', data=concatenated_1, style='dataset',
                    palette=['blue', 'orange'], hue='dataset', legend=True)
    sns.scatterplot(ax=axes[1], x='Experiments', y='Mean of knn changes', data=concatenated_2, style='dataset',
                    palette=['blue', 'orange'], hue='dataset', legend=False)
    sns.scatterplot(ax=axes[2], x='Experiments', y='Mean of knn changes', data=concatenated_3, style='dataset',
                    palette=['blue', 'orange'], hue='dataset', legend=False)
    sns.scatterplot(ax=axes[3], x='Experiments', y='Mean of knn changes', data=concatenated_4, style='dataset',
                    palette=['blue', 'orange'], hue='dataset', legend=False)
    axes[0].set_title('Mean of changes in knns btw. layer {} & {}'.format(layer, layer + 1), fontsize=8)
    axes[1].set_title('Mean of changes in knns btw. layer {} & {}'.format(layer + 1, layer + 2), fontsize=8)
    axes[2].set_title('Mean of changes in knns btw. layer {} & {}'.format(layer + 2, layer + 3), fontsize=8)
    axes[3].set_title('Mean of changes in knns btw. layer {} & {}'.format(layer + 3, layer + 4), fontsize=8)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[3].set_ylabel('')
    # axes[0].legend()
    # axes[3].xlabel("Experiments")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(filepath, bbox_inches="tight")
    plt.show()


def plot_member_non_member_experiments_histogram(sum_similarities_knns_member_all_experiment, sum_similarities_knns_non_member_all_experiment, train_accuracy, test_accuracy, bins, filepath):
    """
    Plot histogram of sum of similar knns of member and non-member data.
    This function is specialised for only this use and has to be generalised to be able to be used more broadly. #TODO

    :param sum_similarities_knns_member_all_experiment:
    :param sum_similarities_knns_non_member_all_experiment:
    :param bins:
    :param filepath:
    :return:
    """
    plt.hist(sum_similarities_knns_member_all_experiment, bins = bins, alpha=0.5, label="Member", linewidth=1.2, edgecolor='black')
    plt.hist(sum_similarities_knns_non_member_all_experiment, bins = bins, alpha=0.5, label="Non-Member", linewidth=1.2, edgecolor='black')
    plt.plot()
    #plt.plot(x_non_member, y_non_member_1, linestyle='None')  # '-o' to connect dots if wanted

    plt.xlabel("Sum of consistent knns (= nns, that stay knns in all layers) (sum for all points and for all layers)")
    plt.ylabel("Amount of experiments")
    plt.suptitle("Distribution of sum of consistent knns")
    plt.title("Model: Lenet-5, Train Acc: {}, Test Acc: {}".format(round(train_accuracy[1], ndigits=4),
                                                                   round(test_accuracy[1], ndigits=4)), fontsize=10)
    plt.legend()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.clf()
