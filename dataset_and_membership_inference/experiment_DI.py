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
from cleverhans_dataset_inference.src.p_value_IF import (
    get_p,
    get_max_p_value,
    ttest_ind_from_stats,
    ttest_ind,
    get_p_values,
    get_fischer,
)
from cleverhans_dataset_inference.src.generate_features_MIA_IF import (
    feature_extractor_MIA,
)
import seaborn as sns
import pandas as pd
from scipy.stats import hmean
import matplotlib.pyplot as plt


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


# generate features
# feature_extractor_MIA()


# train regression model
# code base source: https://github.com/cleverhans-lab/dataset-inference/blob/main/src/notebooks/CIFAR10_mingd.ipynb
split_index = 500

#get distances of test and train set
test_distance = torch.load(
    "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt"
)
train_distance = torch.load(
    "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"
)

#mean distance of training data (sum distances/ amount distances)
#for each distance type (linf, l2, l1)
mean_distance_train = train_distance.mean(dim=(0, 1))
#standard deviation of train distances (mean distances --> deviation of all distances from mean ^2 (to make it positive) /amount distances)
#for each distance type (linf, l2, l1)
std_deviation_distance_train = train_distance.std(dim=(0, 1))

#Sorts the elements of the distance tensor along dim 1 in ascending order by value.
#inside each array of the given dimension the values get sorted, so since dim = 1 and tensor has shape (1000,10,3) --> the distances for each class for one point get sorted
#--> distances are sorted vertically, classes are irrelevant: means for one element in array of first class, first spot is the lowest distance of distance type 0 and so on (distance type 0 can be found: [0][0], [1][0],[2][0],...)
train_distance_sorted_distance_vertically, _ = train_distance.sort(dim=1)
test_distance_sorted_distance_vertically, _ = test_distance.sort(dim=1)

#normalize data
train_distance_sorted_distance_vertically_normalized = (train_distance_sorted_distance_vertically - mean_distance_train) / std_deviation_distance_train
test_distance_sorted_distance_vertically_normalized  = (test_distance_sorted_distance_vertically - mean_distance_train) / std_deviation_distance_train

amount_distances_element = 30

#transpose matrix (1000,10,3) --> (3,10,1000) --> all sorted distances from one distance type are horizontally next to each other
#reshape so that every elements has a list of all 30 distances (3 distances, 10 classes)
#[:,:x] --> for first dimension get all elements (get all 1000 points), for the second all until x (in this case same as getting all distances since x = 30)
train_distance_sorted_distance_types = train_distance_sorted_distance_vertically_normalized.T.reshape(1000, amount_distances_element)[:, :amount_distances_element]
tests_distance_sorted_distance_types = test_distance_sorted_distance_vertically_normalized.T.reshape(1000, amount_distances_element)[:, :amount_distances_element]
#distances are now sorted in one array for each point where [d11,d12,...d110,d21,d22,...], so first all sorted distances for all classes for distance type 0, ...
print(train_distance_sorted_distance_types.shape)
print(tests_distance_sorted_distance_types.shape)

#Concatenates the given sequence of tensors in the dimension 0. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
assert train_distance_sorted_distance_types.shape == tests_distance_sorted_distance_types.shape, "Train and test set must have same shape (amount elements, amount distances)"
#take half train and test points (and thus their distances) and concat to one sequence of tensors.
train_data_regressor = torch.cat((train_distance_sorted_distance_types[:split_index], tests_distance_sorted_distance_types[:split_index]), dim=0)
#concat 0 and 1, where 0 means label member and 1 means label non-member, so the labels are accordingly to data in train
train_labels_regressor = torch.cat((torch.zeros(split_index), torch.ones(split_index)), dim=0)

#randperm returns a random permutation of integers from 0 to n - 1
#all indices of data as a tensor with random order
random_order = torch.randperm(train_labels_regressor.shape[0])
#the data and labels get "shuffeled" --> elements are in a random order
train_data_regressor_random_order = train_data_regressor[random_order]
train_labels_regressor_random_order = train_labels_regressor[random_order]
print("Done preparing training data for linear model.")

#define regressor model
model = nn.Sequential(nn.Linear(amount_distances_element, 100), nn.ReLU(), nn.Linear(100, 1), nn.Tanh())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with tqdm(range(1000)) as pbar: #shows progress meter
    for epoch in pbar:
        optimizer.zero_grad()
        #the input of the model is half member data, half non-member data
        inputs = train_data_regressor_random_order
        outputs = model(inputs)
        loss = -1 * ((2 * train_labels_regressor_random_order - 1) * (outputs.squeeze(-1))).mean()
        loss.backward()
        optimizer.step()
        pbar.set_description("loss {}".format(loss.item()))

print("Finished training linear model")

#run member and non-member data through regressor
outputs_regressor_train = model(train_distance_sorted_distance_types)
outputs_regressor_test = model(tests_distance_sorted_distance_types)

import scipy.stats as stats


def print_inference(outputs_train, outputs_test) -> None:
    """
    Print p-value and mean difference of regressor prediction btw. member and non-member data
    :param outputs_train: Output of regressor for member data
    :param outputs_test: Output of regressor for non-member data
    :return: None
    """
    #[:,0] --> [ first_row:last_row , column_0 ]
    #get mean values of predictions (how close to prediction margin ?) #TODO
    mean_prediction_test, mean_prediction_train = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()
    #get p value (calculated through T-test)
    p_value = get_p(outputs_train, outputs_test)
    print(f"p-value = {p_value} \t| Mean difference of regressor prediction btw. member and non-member data = {mean_prediction_test-mean_prediction_train}")

#get part of the output
outputs_regressor_train_partly, outputs_regressor_test_partly = outputs_regressor_train[split_index:], outputs_regressor_test[split_index:]
#get p-value and mean difference for member and non-member data
print_inference(outputs_regressor_train_partly, outputs_regressor_test_partly)

v_type = "mingd"
if not os.path.exists(
    "/home/inafen/jupyter_notebooks/dataset_inference/cifar10_mingd.h5"
):
    print("Create dataset")
    total_reps = 40
    max_m = 45
    repitions_get_p = 100

    #list with numbers in range (2,max_m)
    m_list = [x for x in range(2, max_m, 1)]
    p_values_all_threat_models_dict = {}

    p_vals_per_rep_no = {}
    r_pbar = tqdm(range(total_reps), leave=False) #shows progress meter
    for repition_number in r_pbar:
        p_values_harmonic_means = []
        for amount_examples_at_once in m_list:
            #get p values as list for random elements (amount: amount_examples_at_once) from output train, test --> repeat repitions_get_p time
            p_values = get_p_values(amount_examples_at_once, outputs_regressor_train_partly, outputs_regressor_test_partly, repitions_get_p)
            try:
                #calculate harmonic mean of p values (~ generally speaking, means get average)
                harmonic_mean = hmean(p_values)
            except:
                harmonic_mean = 1.0
            #keep harmonic means of list of p values
            p_values_harmonic_means.append(harmonic_mean)
            pbar.set_description(f"{repition_number: 2d} amount_examples_at_once={amount_examples_at_once: 3d}: f{harmonic_mean: 0.5f}")
        r_pbar.set_description(f"repition_number: {repition_number + 1}/{total_reps}")
        p_vals_per_rep_no[repition_number] = p_values_harmonic_means
    #save harmonic means, additional information in dict
    p_values_all_threat_models_dict = p_vals_per_rep_no

    #transform dict to dataframe
    df_list = []
    for name, rep_dict in p_values_all_threat_models_dict.items():
        df = (
            pd.DataFrame(rep_dict)
            .reset_index()
            .assign(amount_examples_at_once=lambda r: r.index + 2)
            .drop(["index"], axis=1)
        )
        df_list.append(
            pd.melt(df, id_vars=["amount_examples_at_once"], var_name="repition_number", value_name="p_value").assign(
                threat_model=name
            )
        )
    results_df = pd.concat(df_list)

    results_df.to_hdf(
        "/home/inafen/jupyter_notebooks/dataset_inference/cifar10_mingd.h5", v_type
    )
    results_df
else:
    print("Load dataset")
    results_df = pd.read_hdf(
        "/home/inafen/jupyter_notebooks/dataset_inference/cifar10_mingd.h5", v_type
    )
print(results_df.head())

# remove duplicates from dataframe
results_df_no_duplicates = results_df[~results_df.index.duplicated()]
plt.clf()
ax = sns.lineplot(x="amount_examples_at_once", y="p_value", data=results_df_no_duplicates)
ax.set_xlabel("Number of samples revealed")
ax.set_ylabel("p-value")
plt.show()
