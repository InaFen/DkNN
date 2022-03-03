"""
Experiment 1: Member data
(Sources for specific code parts see as comments down below in specific code parts, DI based on https://github.com/cleverhans-lab/dataset-inference)

Dataset inference attack, specialized for the question how many member data points are necessary for p-value to recognize them as member data.
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

amount_iterations = 10
data_amounts = [200, 100, 50, 30, 10, 5]
amount_data_from_dataset = (
    max(data_amounts) * amount_iterations
)  # how much data should be used to get distances

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load model or build and save model
# torch code source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

PATH_MODEL_TORCH = "/home/inafen/jupyter_notebooks/dataset_inference/model_torch.pth"

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

    # build model
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    torch.save(net.state_dict(), PATH_MODEL_TORCH)

    # get test accuracy
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )

# get distances via feature extractor or load distances for amount_data_from_dataset elements each
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt"
) or not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"
):
    # generate features for amount_data_from_dataset elements each
    feature_extractor_MIA(num_images=amount_data_from_dataset)
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(
        "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt"
    )
    train_distance = torch.load(
        "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"
    )
else:
    # get distances of test and train set (for amount_data_from_dataset elements each)
    test_distance = torch.load(
        "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt"
    )
    train_distance = torch.load(
        "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"
    )

# train regression model
# code base source: https://github.com/cleverhans-lab/dataset-inference/blob/main/src/notebooks/CIFAR10_mingd.ipynb
split_index = 500

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

# experiment 1: Member and non-member data change each iteration (so stay the same for different data amounts in one iteration)
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_members.pickle"
):
    used_indexes = 0
    max_data_amount = max(data_amounts)
    experiment_1_members = {}
    for iteration in range(amount_iterations):
        experiment_1_members_iteration = {}
        # get data for iteration: use different data every iteration
        outputs_regressor_member_iteration = outputs_regressor_train[
            used_indexes : used_indexes + max_data_amount
        ]
        outputs_regressor_non_member_iteration = outputs_regressor_test[
            used_indexes : used_indexes + max_data_amount
        ]
        used_indexes += max_data_amount
        for data_amount in data_amounts:
            # get data_amount points (point = prediction from regressor for a point)
            outputs_regressor_member = outputs_regressor_member_iteration[:data_amount]
            outputs_regressor_non_member = outputs_regressor_non_member_iteration[
                :data_amount
            ]
            # get p-value. Equal amount of member and non-member data is needed for this
            p_value = get_p(outputs_regressor_member, outputs_regressor_non_member)
            if torch.eq(
                outputs_regressor_member[0], outputs_regressor_non_member[0]
            ):  # TODO (also for following experiments) when values absolutely same --> p value nan
                print("equal values --> p-value will become nan")
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
            experiment_1_members_iteration[
                f"data amount: {data_amount}"
            ] = experiment_1_members_data_amount
        experiment_1_members[f"iteration: {iteration}"] = experiment_1_members_iteration
        with open(
            "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_members.pickle",
            "wb",
        ) as f:
            pickle.dump(experiment_1_members, f)
else:
    # open pickle
    with open(
        "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_members.pickle",
        "rb",
    ) as f:
        experiment_1_members = pickle.load(f)

# experiment 1.1: random data used for each data amount
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_1_members.pickle"
):
    experiment_1_1_members = {}
    outputs_regressor_member_total = outputs_regressor_train.shape[0]
    for iteration in range(amount_iterations):
        experiment_1_1_members_iteration = {} #TODO isnt used down below
        for data_amount in data_amounts:
            # get data_amount points (point = prediction from regressor for a point)
            # get random elements from train and test set
            positions = torch.randperm(outputs_regressor_member_total)[:data_amount]
            outputs_regressor_member = outputs_regressor_train[positions]
            outputs_regressor_non_member = outputs_regressor_test[positions]
            # get p-value. Equal amount of member and non-member data is needed for this
            p_value = get_p(outputs_regressor_member, outputs_regressor_non_member)
            print(f"1.1: Amount of data: {data_amount}| P-value: {p_value}")
            # save in dict
            experiment_1_members_data_amount = {}
            experiment_1_members_data_amount["p value"] = p_value
            experiment_1_members_data_amount[
                "outputs_regressor_member"
            ] = outputs_regressor_member
            experiment_1_members_data_amount[
                "outputs_regressor_non_member"
            ] = outputs_regressor_non_member
            experiment_1_members_iteration[
                f"data amount: {data_amount}"
            ] = experiment_1_members_data_amount
        experiment_1_1_members[
            f"iteration: {iteration}"
        ] = experiment_1_members_iteration
        with open(
            "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_1_members.pickle",
            "wb",
        ) as f:
            pickle.dump(experiment_1_1_members, f)
else:
    # open pickle
    with open(
        "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_1_members.pickle",
        "rb",
    ) as f:
        experiment_1_1_members = pickle.load(f)

# experiment 1.2: Member data change each iteration (so stay the same for different data amounts in one iteration), non-member data random for each data amount
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_2_members.pickle"
):
    used_indexes = 0
    max_data_amount = max(data_amounts)
    outputs_regressor_member_total = outputs_regressor_train.shape[0]
    experiment_1_2_members = {}
    for iteration in range(amount_iterations):
        experiment_1_members_iteration = {}
        # get data for iteration: use different data every iteration
        outputs_regressor_member_iteration = outputs_regressor_train[
            used_indexes : used_indexes + max_data_amount
        ]
        used_indexes += max_data_amount
        for data_amount in data_amounts:
            # get random non member data
            positions = torch.randperm(outputs_regressor_member_total)[:data_amount]

            # get data_amount points (point = prediction from regressor for a point)
            outputs_regressor_member = outputs_regressor_member_iteration[:data_amount]
            outputs_regressor_non_member = outputs_regressor_test[positions]
            # get p-value. Equal amount of member and non-member data is needed for this
            p_value = get_p(outputs_regressor_member, outputs_regressor_non_member)
            print(f"1.2: Amount of data: {data_amount}| P-value: {p_value}")
            # save in dict
            experiment_1_members_data_amount = {}
            experiment_1_members_data_amount["p value"] = p_value
            experiment_1_members_data_amount[
                "outputs_regressor_member"
            ] = outputs_regressor_member
            experiment_1_members_data_amount[
                "outputs_regressor_non_member"
            ] = outputs_regressor_non_member
            experiment_1_members_iteration[
                f"data amount: {data_amount}"
            ] = experiment_1_members_data_amount
        experiment_1_2_members[
            f"iteration: {iteration}"
        ] = experiment_1_members_iteration
        with open(
            "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_2_members.pickle",
            "wb",
        ) as f:
            pickle.dump(experiment_1_2_members, f)
else:
    # open pickle
    with open(
        "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_2_members.pickle",
        "rb",
    ) as f:
        experiment_1_2_members = pickle.load(f)

# experiment 1.3: same members per iteration, always same non-members
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_3_members.pickle"
):
    used_indexes = 0
    max_data_amount = max(data_amounts)
    experiment_1_3_members = {}
    for iteration in range(amount_iterations):
        experiment_1_members_iteration = {}
        # get data for iteration: use different data every iteration
        outputs_regressor_member_iteration = outputs_regressor_train[
            used_indexes : used_indexes + max_data_amount
        ]
        used_indexes += max_data_amount
        for data_amount in data_amounts:
            # get data_amount points (point = prediction from regressor for a point)
            outputs_regressor_member = outputs_regressor_member_iteration[:data_amount]
            outputs_regressor_non_member = outputs_regressor_test[:data_amount]
            # get p-value. Equal amount of member and non-member data is needed for this
            p_value = get_p(outputs_regressor_member, outputs_regressor_non_member)
            print(f"1.3: Amount of data: {data_amount}| P-value: {p_value}")
            # save in dict
            experiment_1_members_data_amount = {}
            experiment_1_members_data_amount["p value"] = p_value
            experiment_1_members_data_amount[
                "outputs_regressor_member"
            ] = outputs_regressor_member
            experiment_1_members_data_amount[
                "outputs_regressor_non_member"
            ] = outputs_regressor_non_member
            experiment_1_members_iteration[
                f"data amount: {data_amount}"
            ] = experiment_1_members_data_amount
        experiment_1_3_members[
            f"iteration: {iteration}"
        ] = experiment_1_members_iteration
        with open(
            "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_3_members.pickle",
            "wb",
        ) as f:
            pickle.dump(experiment_1_members, f)
else:
    # open pickle
    with open(
        "/home/inafen/jupyter_notebooks/dataset_and_membership_inference/experiment_1_3_members.pickle",
        "rb",
    ) as f:
        experiment_1_3_members = pickle.load(f)


def plot_p_value(amounts_data, p_values, title):
    """
    Plot p-values, adapted for experiment 1 (= similar to plot_p_value_exp2 besides amounts_data and x label)

    :param amount_members: amounts of members in "member" data used for calculation of p-value
    :param p_values: p-values
    :param title: Title of plot
    :return: None
    """
    data = {"amounts_data": amounts_data, "p_values": p_values}
    dataframe = pd.DataFrame(data)

    plt.clf()
    ax = sns.lineplot(x="amounts_data", y="p_values", data=dataframe)
    ax.set_xlabel("Number of samples revealed")
    ax.set_ylabel("p-value")
    ax.set_title(title)
    plt.show()


# plot iteration individually
def plot_iteration(experiment, title: str) -> None:
    """
    Plot p-values for one iteration, slightly adapted for experiment 1.

    :param experiment: dataframe with experiment information
    :param title: title of plot
    :return: None
    """
    for iteration in range(amount_iterations):
        p_values_iteration = []
        for data_amount in data_amounts:
            p_values_iteration.append(
                experiment[f"iteration: {iteration}"][f"data amount: {data_amount}"][
                    "p value"
                ]
            )
        plot_p_value(
            amounts_data=data_amounts, p_values=p_values_iteration, title=title
        )


# plot iterations together
def plot_mean_all_iterations(experiment, title: str) -> None:
    """
    Plot mean of p-values for all iterations, slightly adapted for experiment 1.

    :param experiment: dataframe with experiment information
    :param title: title of plot
    :return: None
    """
    p_values = [[] for i in range(len(data_amounts))]
    for iteration in range(amount_iterations):
        p_values_iteration = []
        for data_amount in data_amounts:
            p_values_iteration.append(
                experiment[f"iteration: {iteration}"][f"data amount: {data_amount}"][
                    "p value"
                ]
            )
        for data_amount_index in range(len(data_amounts)):
            p_values[data_amount_index].append(p_values_iteration[data_amount_index])
    # get mean of p values
    p_values_mean = []
    for data_amount_index in range(len(data_amounts)):
        p_values_mean.append(mean(p_values[data_amount_index]))
    plot_p_value(amounts_data=data_amounts, p_values=p_values_mean, title=title)


# plot experiments
plot_mean_all_iterations(
    experiment_1_members, title="Exp. 1: Same members, non-members per iteration"
)
plot_mean_all_iterations(
    experiment_1_1_members,
    title="Exp. 1.1: Random members, non-members for each data amount",
)
plot_mean_all_iterations(
    experiment_1_2_members,
    title="Exp. 1.2: Same members per iteration, random non-members for each data amount",
)
plot_mean_all_iterations(
    experiment_1_3_members,
    title="Exp. 1.3: Same members per iteration, always same non-members",
)
