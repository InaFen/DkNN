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

import timeit

start = timeit.default_timer()

amount_repetitions = 10

#for experiment 1
data_amounts = [200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

#experiment 2
amount_members = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
amount_data_from_dataset = (
    600
)  # how much data should be used to get distances, double amount_data because non-members have to be split (see down below)
amount_data = 300  # how much data should be used for p-value , for each dataset #TODO change to smaller amount to have more random options
#TODO check standard deviation at 200 --> negative??

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load model or load pretrained model and save model

PATH_MODEL_TORCH = "/home/inafen/jupyter_notebooks/dataset_inference/mobilenet_v2.pt"
PATH_TRAIN_VULNERABILITY = "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulnerability_mobilenet_v2_600.pt"
PATH_TEST_VULNERABILITY = "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulnerability_mobilenet_v2_600.pt"
PATH_EXPERIMENT_1 = "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_members_v2.pickle"
PATH_EXPERIMENT_2 = "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_2_(non)_members_all_10_perecentage_v2.pickle"

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

    #get pretrained model
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
    #Accuracy of the network on the 10000 test images: 88 %

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
    #Accuracy of the network on the 50000 train images: 93 %

# get distances via feature extractor or load distances for amount_data_from_dataset elements each
if not os.path.exists(
    PATH_TRAIN_VULNERABILITY
) or not os.path.exists(
    PATH_TEST_VULNERABILITY
):
    # generate features for amount_data_from_dataset elements each
    feature_extractor_MIA(num_images=amount_data_from_dataset, test_distance_path=PATH_TEST_VULNERABILITY, train_distance_path=PATH_TRAIN_VULNERABILITY, model_path=PATH_MODEL_TORCH)
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(
        PATH_TEST_VULNERABILITY
    )
    train_distance = torch.load(
        PATH_TRAIN_VULNERABILITY
    )
else:
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(
        PATH_TEST_VULNERABILITY
    )
    train_distance = torch.load(
        PATH_TRAIN_VULNERABILITY
    )

# train regression model
# code base source: https://github.com/cleverhans-lab/dataset-inference/blob/main/src/notebooks/CIFAR10_mingd.ipynb
split_index = 100 #TODO change to 500

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

# experiment 1: random data used for each data amount
if not os.path.exists(
    PATH_EXPERIMENT_1
):
    experiment_1_members = {}
    outputs_regressor_member_total = outputs_regressor_train.shape[0]
    for repetition in range(amount_repetitions):
        experiment_1_members_repetition = {}
        for data_amount in data_amounts:
            # get data_amount points (point = prediction from regressor for a point)
            # get random elements from train and test set
            positions = torch.randperm(outputs_regressor_member_total)[:data_amount]
            outputs_regressor_member = outputs_regressor_train[positions]
            outputs_regressor_non_member = outputs_regressor_test[positions]
            # get p-value. Equal amount of member and non-member data is needed for this
            p_value = get_p(outputs_regressor_member, outputs_regressor_non_member)
            print(f"1: Amount of data: {data_amount}| P-value: {p_value}")
            # save in dict
            experiment_1_members_data_amount = {}
            experiment_1_members_data_amount["p value"] = p_value
            experiment_1_members_data_amount[
                "outputs_regressor_member"
            ] = outputs_regressor_member
            experiment_1_members_data_amount[
                "outputs_regressor_non_member"
            ] = outputs_regressor_non_member
            experiment_1_members_repetition[
                f"data amount: {data_amount}"
            ] = experiment_1_members_data_amount
        experiment_1_members[
            f"repetition: {repetition}"
        ] = experiment_1_members_repetition
        with open(
            PATH_EXPERIMENT_1,
            "wb",
        ) as f:
            pickle.dump(experiment_1_members, f)
else:
    # open pickle
    with open(
        PATH_EXPERIMENT_1,
        "rb",
    ) as f:
        experiment_1_members = pickle.load(f)

# split outputs_regressor_test in test data (non-member) used for p-value & test data used as part of "member" data used for p-value
outputs_regressor_test_non_member, outputs_regressor_test_member_fake = torch.split(
    outputs_regressor_test, int(amount_data_from_dataset//2)
)


if not os.path.exists(
    PATH_EXPERIMENT_2
):
    experiment_2_members = {}
    outputs_regressor_member_fake_total = outputs_regressor_test_member_fake.shape[0]
    for repetition in range(amount_repetitions):
        experiment_2_members_repetition = {}
        for percentage_members in amount_members:
            # get random indexes for non member data
            positions_non_member = torch.randperm(outputs_regressor_member_fake_total)[
                :amount_data
            ]
            # get random indexes for member data. Percentage members* amount data members are used.
            positions_member = torch.randperm(outputs_regressor_member_fake_total)[
                : int(amount_data * percentage_members)
            ]
            # get random indexes for non-members in "member" data.
            positions_member_fake = torch.randperm(outputs_regressor_member_fake_total)[
                : int(amount_data * (1 - percentage_members))
            ]
            # get random data sets. Create member set out of non-members and members
            outputs_regressor_non_member = outputs_regressor_test_non_member[
                positions_non_member
            ]
            outputs_regressor_member = outputs_regressor_train[positions_member]
            outputs_regressor_member_fake = outputs_regressor_test_member_fake[
                positions_member_fake
            ]
            # Create member set out of non-members and members
            outputs_regressor_member_and_fake_member = torch.cat(
                (outputs_regressor_member, outputs_regressor_member_fake), 0
            ) #TODO why 299 instead of 300 elements?
            if torch.eq(
                outputs_regressor_member[0], outputs_regressor_non_member[0]
            ):  # TODO (also for following experiments) when values absolutely same --> p value nan
                print("equal values --> p-value will become nan")
            # get p-value. Equal amount of "member" and non-member data is needed for this
            p_value = get_p(
                outputs_regressor_member_and_fake_member, outputs_regressor_non_member
            )
            print(f"1: Perecentage members: {percentage_members}| P-value: {p_value}")
            # save in dict
            experiment_2_members_percentage_member = {}
            experiment_2_members_percentage_member["p value"] = p_value
            experiment_2_members_percentage_member[
                "outputs_regressor_member"
            ] = outputs_regressor_member
            experiment_2_members_percentage_member[
                "outputs_regressor_non_member"
            ] = outputs_regressor_non_member
            experiment_2_members_repetition[
                f"percentage members: {percentage_members}"
            ] = experiment_2_members_percentage_member
        experiment_2_members[f"repetition: {repetition}"] = experiment_2_members_repetition
        with open(
            PATH_EXPERIMENT_2,
            "wb",
        ) as f:
            pickle.dump(experiment_2_members, f)
else:
    # open pickle
    with open(
        PATH_EXPERIMENT_2,
        "rb",
    ) as f:
        experiment_2_members = pickle.load(f)


def plot_p_value( p_values, title: str, name_x_column: str, data_x_column, x_label:str ):
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
    ax = sns.lineplot(x=name_x_column, y="p_values", data=dataframe, marker = 'o', ci = "sd")
    ax.set_xlabel(x_label)
    ax.set_ylabel("p-value")
    ax.set_title(title)
    #plt.text(-38, -0.38,
    #         "Dataset Inference attack on member sets, mean of p-values over 10 repetitions \nModel information: MobileNetV2 | Train accuracy: 93 %  | Test accuracy: 88 % \nP-value calculation: Amount of random elements in mixed and non-member set each: 300 \n ",
    #         fontsize=10, bbox={'facecolor': '.9', 'boxstyle': 'square', 'edgecolor': '.9'})
    #plt.subplots_adjust(bottom=0.3)
    plt.text(-0.075, -0.0007,
             "Dataset Inference attack on mixed (members and non-members) sets\n--> mean of p-values over 10 repetitions \nModel information: MobileNetV2 | Train accuracy: 93 %  | Test accuracy: 88 % \nP-value calculation: Amount of random elements in mixed and non-member set each: 300 \n ",
             fontsize=10, bbox={'facecolor': '.9', 'boxstyle': 'square', 'edgecolor': '.9'})
    plt.subplots_adjust(bottom=0.3)
    plt.savefig("/home/inafen/jupyter_notebooks/dataset_and_membership_inference/exp_2_mean.png")
    plt.show()



# plot repetition individually
def plot_repetition_in_one_graph(experiment, title: str, name_x_column: str, amounts, x_label: str) -> None:
    """
    Plot p-values for all repetitions in one graph

    :param experiment: dataframe with experiment information
    :param title: title of plot
    :param name_x_column: Name of df column which values will be used for x-axis (e.g. data amount, percentage members)
    :param amounts: Data of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param x_label: Label for x-axis (e.g. "Number of samples revealed", "Percentage of members")
    :return: None
    """
    data_x_column = []
    p_values = []
    repetitions = []
    for repetition in range(amount_repetitions):
        for amount in amounts:
            p_values.append(
                experiment[f"repetition: {repetition}"][f"{name_x_column}: {amount}"][
                    "p value"
                ]

            )
            data_x_column.append(amount)
            repetitions.append(repetition)

    data = {name_x_column: data_x_column, "p_values": p_values, "repetitions": repetitions}
    dataframe = pd.DataFrame(data)
    print(dataframe.head())
    plt.clf()
    ax = sns.lineplot(x=name_x_column, y="p_values", data=dataframe, marker='o', hue = "repetitions", palette = "hls")
    ax.set_xlabel(x_label)
    ax.set_ylabel("p-value")
    ax.set_title(title)
    plt.text(-0.08,-0.001, "Dataset Inference attack on mixed (members and non-members) sets \nModel information: MobileNetV2 | Train accuracy: 93 %  | Test accuracy: 88 % \nP-value calculation: Amount of random elements in mixed and non-member set each: 300 \n ", fontsize = 10, bbox={'facecolor': '.9', 'boxstyle':'square', 'edgecolor': '.9'} )
    plt.subplots_adjust(bottom=0.3)
    plt.savefig("/home/inafen/jupyter_notebooks/dataset_and_membership_inference/exp_2_all_repetitions.png")
    plt.show()


# plot repetitions together
def plot_mean_all_repetitions(experiment, title: str, amount_repetitions: int, name_x_column: str, amounts, x_label:str) -> None:
    """
    Plot mean of p-values for all repetitions.

    :param experiment: dataframe with experiment information
    :param title: title of plot
    :param amount_repetitions: how many times the experiment was repeated
    :param name_x_column: Name of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param data_amounts: Data of df column which values will be used for x-axis (e.g. amounts_data, amounts_member)
    :param x_label: Label for x-axis (e.g. "Number of samples revealed", "Percentage of members")
    :return: None
    """
    p_values = []
    data_x_column = []
    for repetition in range(amount_repetitions):
        for amount in amounts:
            p_values.append(
                experiment[f"repetition: {repetition}"][f"{name_x_column}: {amount}"][
                    "p value"
                ])
            data_x_column.append(amount)
    #mean gets plot automatically for each category of x column values (so e.g. for all 0.1 % of members --> overall mean is calculated)
    plot_p_value( p_values=p_values, title=title, name_x_column = name_x_column, data_x_column=data_x_column, x_label=x_label)



#plot experiments
#plot_mean_all_repetitions(
#    experiment_1_members, title="Exp. 1, mean over 10 repetitions", amount_repetitions= amount_repetitions, name_x_column="data amount", amounts=data_amounts, x_label="Number of samples revealed"
#)

#plot_repetition_in_one_graph(
#    experiment_2_members, title="Exp. 2, all repetitions plotted separately", name_x_column="percentage members", amounts=amount_members, x_label="Percentage of members in mixed set"
#)  #: Random non-members, percentage of random members and non-members as <<members>>")
plot_mean_all_repetitions(
    experiment_2_members, title="Exp. 2, mean over 10 repetitions", amount_repetitions= amount_repetitions, name_x_column="percentage members", amounts=amount_members, x_label="Percentage of members"
)  #: Random non-members, percentage of random members and non-members as <<members>>")


stop = timeit.default_timer()

print('Time: ', stop - start)