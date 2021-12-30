import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange


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


def plot_mean_knns_layer(knns_attribute, filepath, differences_knns_total, ylabel, xlabel, title):
    """
    Creates bar graph of mean of knns for each layer for one data set.

    :param knns_attribute: indices or labels of knns
    :param filepath: filepath to where plot should be saved.
    :param differences_knns_total: amount of knns changes for data per layer
    """

    # get mean of knns per layer
    mean_knns_layers = []
    for layer in range(1, len(knns_attribute)):
        mean_knns_layers.append(np.mean(differences_knns_total.get("layer {}".format(layer))))

    # plot mean

    layers = []
    for layer in range(1, len(knns_attribute)):
        layers.append("layer {}".format(layer))
    x_pos = np.arange(len(layers))
    plt.bar(x_pos, mean_knns_layers)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_pos,layers)
    plt.show()

    plt.savefig(filepath, bbox_inches="tight")


def plot_changes_knns_3(mean_layers_1, mean_layers_2, mean_layers_3, knns_attribute, filepath, xlabel, ylabel, title,
                        label_1='Train', label_2='Test', label_3='Noisy'):
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