"""
Experiment 1 and 2
(Sources for specific code parts see as comments down below in specific code parts, DI based on https://github.com/cleverhans-lab/dataset-inference)

"""


from models_torch import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from os import path
import os
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
import random
from cleverhans_dataset_inference.src.p_value_IF import get_p
from cleverhans_dataset_inference.src.generate_features_MIA_IF import (
    feature_extractor_MIA,
)
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from models_torch_cifar import mobilenet_v2

amount_repetitions = 10


# experiment 3.1
amount_members = 25  # smallest size of member dataset which gets recognized as such by DI #TODO change t0 45?
percentage_members_in_mixed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

amount_data_from_dataset = 600  # all available data --> how much data should be used to get distances, double amount_data because non-members have to be split (see down below) #TODO change back to 400*amount_repitions
amount_data = 250  # how much data should be used for p-value , for each dataset

# TODO depending on seed success of experiments changes
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load model or load pretrained model and save model

PATH_MODEL_TORCH = "/home/inafen/jupyter_notebooks/dataset_inference/mobilenet_v2.pt"
PATH_TRAIN_VULNERABILITY = "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulnerability_mobilenet_v2_600.pt"
PATH_TEST_VULNERABILITY = "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulnerability_mobilenet_v2_600.pt"
PATH_EXPERIMENT_3 = "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_3.pickle"
# TODO "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_2_(non)_members.pickle" --> different results
if not (path.exists(PATH_MODEL_TORCH)):
    # get train and test data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # get pretrained model
    model = mobilenet_v2(pretrained=True)
    model.eval()  # for evaluation --> if model was to be trained more, switch back to model.train()

    torch.save(model.state_dict(), PATH_MODEL_TORCH)

    # torch code source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    # get test accuracy
    print("Get test accuracy")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
    # Accuracy of the network on the 10000 test images: 88 %

    print("Get train accuracy")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 50000 train images: {100 * correct // total} %"
    )
    # Accuracy of the network on the 50000 train images: 93 %

# get distances via feature extractor or load distances for amount_data_from_dataset elements each
if not os.path.exists(PATH_TRAIN_VULNERABILITY) or not os.path.exists(
    PATH_TEST_VULNERABILITY
):
    # generate features for amount_data_from_dataset elements each
    feature_extractor_MIA(
        num_images=amount_data_from_dataset,
        test_distance_path=PATH_TEST_VULNERABILITY,
        train_distance_path=PATH_TRAIN_VULNERABILITY,
        model_path=PATH_MODEL_TORCH,
    )
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(PATH_TEST_VULNERABILITY)
    train_distance = torch.load(PATH_TRAIN_VULNERABILITY)
else:
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(PATH_TEST_VULNERABILITY)
    train_distance = torch.load(PATH_TRAIN_VULNERABILITY)

# train regression model
# code base source: https://github.com/cleverhans-lab/dataset-inference/blob/main/src/notebooks/CIFAR10_mingd.ipynb
split_index = 100  # TODO 500

# mean distance of training data (sum distances/ amount distances)
# for each distance type (linf, l2, l1)
mean_distance_train = train_distance.mean(dim=(0, 1))
# standard deviation of train distances (mean distances --> deviation of all distances from mean ^2 (to make it positive) /amount distances)
# for each distance type (linf, l2, l1)
std_deviation_distance_train = train_distance.std(dim=(0, 1))

# Sort: sorts the elements of the distance tensor along dim 1 in ascending order by value.
# inside each array of the given dimension the values get sorted, so since dim = 1 and tensor has shape (amount_data_from_dataset,10,3) --> the distances for each class for one point get sorted
# --> distances are sorted vertically, classes are irrelevant: means for one element in array of first class, first spot is the lowest distance of distance type 0 and so on (distance type 0 can be found: [0][0], [1][0],[2][0],...)
train_distance_sorted_distance_vertically, _ = train_distance.sort(dim=1)
test_distance_sorted_distance_vertically, _ = test_distance.sort(dim=1)

# normalize data
train_distance_sorted_distance_vertically_normalized = (
    train_distance_sorted_distance_vertically - mean_distance_train
) / std_deviation_distance_train
test_distance_sorted_distance_vertically_normalized = (
    test_distance_sorted_distance_vertically - mean_distance_train
) / std_deviation_distance_train

amount_distances_element = 30

# transpose matrix (amount_data_from_dataset,10,3) --> (3,10,amount_data_from_dataset) --> all sorted distances from one distance type are horizontally next to each other
# reshape so that every elements has a list of all 30 distances (3 distances, 10 classes)
# [:,:x] --> for first dimension get all elements (get all amount_data_from_dataset points), for the second all until x (in this case same as getting all distances since x = 30)
train_distance_sorted_distance_types = (
    train_distance_sorted_distance_vertically_normalized.T.reshape(
        amount_data_from_dataset, amount_distances_element
    )[:, :amount_distances_element]
)
tests_distance_sorted_distance_types = (
    test_distance_sorted_distance_vertically_normalized.T.reshape(
        amount_data_from_dataset, amount_distances_element
    )[:, :amount_distances_element]
)
# distances are now sorted in one array for each point where [d11,d12,...d110,d21,d22,...], so first all sorted distances for all classes for distance type 0, ...
print(train_distance_sorted_distance_types.shape)
print(tests_distance_sorted_distance_types.shape)

# Concatenates the given sequence of tensors in the dimension 0. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
assert (
    train_distance_sorted_distance_types.shape
    == tests_distance_sorted_distance_types.shape
), "Train and test set must have same shape (amount elements, amount distances)"
# take half train and test points (and thus their distances) and concat to one sequence of tensors.
train_data_regressor = torch.cat(
    (
        train_distance_sorted_distance_types[:split_index],
        tests_distance_sorted_distance_types[:split_index],
    ),
    dim=0,
)
# concat 0 and 1, where 0 means label member and 1 means label non-member, so the labels are accordingly to data in train
train_labels_regressor = torch.cat(
    (torch.zeros(split_index), torch.ones(split_index)), dim=0
)

# randperm returns a random permutation of integers from 0 to n - 1
# all indices of data as a tensor with random order
random_order = torch.randperm(train_labels_regressor.shape[0])
# the data and labels get "shuffeled" --> elements are in a random order
train_data_regressor_random_order = train_data_regressor[random_order]
train_labels_regressor_random_order = train_labels_regressor[random_order]
print("Done preparing training data for linear model.")

# define regressor model
model = nn.Sequential(
    nn.Linear(amount_distances_element, 100), nn.ReLU(), nn.Linear(100, 1), nn.Tanh()
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with tqdm(range(1000)) as pbar:  # shows progress meter
    for epoch in pbar:
        optimizer.zero_grad()
        # the input of the model is half member data, half non-member data
        inputs = train_data_regressor_random_order
        outputs = model(inputs)
        loss = (
            -1
            * (
                (2 * train_labels_regressor_random_order - 1) * (outputs.squeeze(-1))
            ).mean()
        )
        loss.backward()
        optimizer.step()
        pbar.set_description("loss {}".format(loss.item()))

print("Finished training linear model")

# run member and non-member data through regressor
# outputs are how likely data point is member
outputs_regressor_train = model(train_distance_sorted_distance_types)
outputs_regressor_test = model(tests_distance_sorted_distance_types)
# split outputs_regressor_test in test data (non-member) used for p-value & test data used as part of "member" data used for p-value
outputs_regressor_test_non_member, outputs_regressor_test_member_fake = torch.split(
    outputs_regressor_test, int(amount_data_from_dataset // 2)
)


def get_p_value_without_one_point_for_whole_range(
    mixed_set, non_member_set, start_index: int, end_index: int
):
    """
    Remove one element from mixed set (and thus also one element from non-members) --> calculate p-value without this element.
    Repeat for all elements in given index range. Put the element back in mixed set after p-value is calculated.

    :param mixed_set: Dataset with members and non-members (output of regressor)
    :param non_members: Dataset with non-members (output of regressor)
    :param start_index: Index of first element that should be removed
    :param end_index: Index of last element that should be removed
    :return: p-values (dict with all p-values from all iterations)
    """
    p_values_and_index = {}  # TODO remove in case it is not needed later
    p_values = []
    for index in range(start_index, end_index + 1):
        # remove element from mixed set
        mixed_set_without_element = torch.cat(
            [mixed_set[:index], mixed_set[index + 1 :]]
        )
        # remove also this element from non-members so has same shape. Could remove any element but to not always remove same element in each loop, remove the current_index element
        non_members_without_element = torch.cat(
            [non_member_set[:index], non_member_set[index + 1 :]]
        )
        # get p-value
        p_value_without_element = get_p(
            mixed_set_without_element, non_members_without_element
        )
        p_values_and_index[
            "Index of removed element {}".format(index)
        ] = p_value_without_element
        p_values.append(p_value_without_element)
    return p_values


# experiment 3.1
# first experiment
if not os.path.exists(PATH_EXPERIMENT_3):
    experiment = {}
    # go through all percentages of members in mixed set
    for percentage_members in percentage_members_in_mixed:
        current_values = {}
        # how many members, non-members should be in mixed set
        amount_members_in_mixed = int(amount_data * percentage_members)
        amount_non_members_in_mixed = int(amount_data * (1 - percentage_members))
        # get random indexes for elements for mixed and non-member set, according to how many member & non-member elements should be in mixed
        # mixed
        positions_members_in_mixed = torch.randperm(amount_data)[
            :amount_members_in_mixed
        ]
        positions_non_members_in_mixed = torch.randperm(amount_data)[
            :amount_non_members_in_mixed
        ]
        # non-member
        positions_non_members = torch.randperm(amount_data)[:amount_data]
        # get actual elements
        members_in_mixed = outputs_regressor_train[positions_members_in_mixed]
        non_members_in_mixed = outputs_regressor_test_member_fake[
            positions_non_members_in_mixed
        ]
        non_members = outputs_regressor_test[positions_non_members]
        # create mixed set
        mixed_set = torch.cat((members_in_mixed, non_members_in_mixed), dim=0)
        # remove non_member from mixed set --> get p-value --> put non-member back in
        # repeat for each non-member
        # first non-member is at index amount_members_in_mixed+1 in mixed set and last non-member is the last index, so equals amount_data
        p_values_removed_non_members = get_p_value_without_one_point_for_whole_range(
            mixed_set,
            non_members,
            start_index=amount_members_in_mixed + 1,
            end_index=amount_data,
        )
        current_values["p-values removed non-members"] = p_values_removed_non_members
        # repeat for removing members
        p_values_removed_members = get_p_value_without_one_point_for_whole_range(
            mixed_set, non_members, start_index=0, end_index=amount_non_members_in_mixed
        )
        current_values["p-values removed members"] = p_values_removed_members
        experiment[
            "Percentage members in mixed set {}".format(percentage_members)
        ] = current_values
    with open(
        PATH_EXPERIMENT_3,
        "wb",
    ) as f:
        pickle.dump(experiment, f)
else:
    # open pickle
    with open(
        PATH_EXPERIMENT_3,
        "rb",
    ) as f:
        experiment_3_1 = pickle.load(f)


def plot_p_value(p_values, title: str, name_x_column: str, data_x_column, x_label: str):
    """
    Plot p-values as lineplot

    :param p_values: p-values
    :param title: Title of plot
    :param name_x_column: Name of df column which values will be used for x-axis (e.g. data amount, percentage members)
    :param data_x_column: Data of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param x_label: Label for x-axis (e.g. "Number of samples revealed", "Percentage of members")
    :return: None
    """
    data = {name_x_column: data_x_column, "p_values": p_values}
    dataframe = pd.DataFrame(data)

    plt.clf()
    ax = sns.lineplot(
        x=name_x_column, y="p_values", data=dataframe, marker="o", ci="sd"
    )
    # if line is not wanted for e.g. experiment 1: change to scatterplot TODO
    ax.set_xlabel(x_label)
    ax.set_ylabel("p-value")
    ax.set_title(title)
    plt.show()


# plot repetitions together
def plot_mean_p_values_all_percentages(
    experiment,
    title: str,
    percentage_members_in_mixed,
    name_x_column: str,
    x_label: str,
) -> None:
    """
    Plot mean of p-values for all distributions of members in mixed set.
    Plot mean for extracted members and non-members seperately

    :param experiment: dataframe with experiment information
    :param title: title of plot
    :param amount_repetitions: how many times the experiment was repeated
    :param name_x_column: Name of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param data_amounts: Data of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param x_label: Label for x-axis (e.g. "Number of samples revealed", "Percentage of members")
    :return: None
    """
    p_values = []
    data_x_column_member = []
    data_x_column_non_member = []
    p_values_removed_members = []
    p_values_removed_non_members = []
    for percentage_member in percentage_members_in_mixed:
        # TODO why one p-value more for members?
        # append all the elements from the lists
        for p_value in experiment[
            "Percentage members in mixed set {}".format(percentage_member)
        ]["p-values removed members"]:
            p_values_removed_members.append(p_value)
            data_x_column_member.append(percentage_member)
        for p_value in experiment[
            "Percentage members in mixed set {}".format(percentage_member)
        ]["p-values removed non-members"]:
            p_values_removed_non_members.append(p_value)
            data_x_column_non_member.append(percentage_member)
    # mean gets plot automatically for each category of x column values (so e.g. for all 0.1 % of members --> overall mean is calculated)
    plot_p_value(
        p_values=p_values_removed_members,
        title="Exp. 3.1.1, removal of members",
        name_x_column=name_x_column,
        data_x_column=data_x_column_member,
        x_label=x_label,
    )
    plot_p_value(
        p_values=p_values_removed_non_members,
        title="Exp. 3.1.1, removal of non-members",
        name_x_column=name_x_column,
        data_x_column=data_x_column_non_member,
        x_label=x_label,
    )


# TODO why goes one graph only until 0.9?
plot_mean_p_values_all_percentages(
    experiment=experiment_3_1,
    title="Exp. 3.1.1",
    percentage_members_in_mixed=percentage_members_in_mixed,
    name_x_column="Percentage of members in mixed set",
    x_label="Percentage of members in mixed set",
)
