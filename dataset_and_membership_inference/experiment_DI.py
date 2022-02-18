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

test_distance = torch.load(
    "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt"
)
train_distance = torch.load(
    "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"
)
mean_cifar = train_distance.mean(dim=(0, 1))
std_cifar = train_distance.std(dim=(0, 1))

train_distance = train_distance.sort(dim=1)[0]
test_distance = test_distance.sort(dim=1)[0]

train_distance = (train_distance - mean_cifar) / std_cifar
test_distance = (test_distance - mean_cifar) / std_cifar

f_num = 30
a_num = 30

trains_n = train_distance.T.reshape(1000, f_num)[:, :a_num]
tests_n = test_distance.T.reshape(1000, f_num)[:, :a_num]
print(trains_n.shape)

n_ex = split_index
train = torch.cat((trains_n[:n_ex], tests_n[:n_ex]), dim=0)
y = torch.cat((torch.zeros(n_ex), torch.ones(n_ex)), dim=0)

rand = torch.randperm(y.shape[0])
train = train[rand]
y = y[rand]

model = nn.Sequential(nn.Linear(a_num, 100), nn.ReLU(), nn.Linear(100, 1), nn.Tanh())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with tqdm(range(1000)) as pbar:
    for epoch in pbar:
        optimizer.zero_grad()
        inputs = train
        outputs = model(inputs)
        loss = -1 * ((2 * y - 1) * (outputs.squeeze(-1))).mean()
        loss.backward()
        optimizer.step()
        pbar.set_description("loss {}".format(loss.item()))

print("Finished training linear model")

outputs_tr = model(trains_n)
outputs_te = model(tests_n)

import scipy.stats as stats


def print_inference(outputs_train, outputs_test):
    m1, m2 = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()
    pval = get_p(outputs_train, outputs_test)
    print(f"p-value = {pval} \t| Mean difference = {m1-m2}")


outputs_tr, outputs_te = outputs_tr[split_index:], outputs_te[split_index:]

print_inference(outputs_tr, outputs_te)

v_type = "mingd"
if not os.path.exists(
    f"/home/inafen/jupyter_notebooks/dataset_inference/cifar10_{v_type}.h5"
):
    total_reps = 40
    max_m = 45
    total_inner_rep = 100

    m_list = [x for x in range(2, max_m, 1)]
    p_values_all_threat_models_dict = {}

    p_vals_per_rep_no = {}
    r_pbar = tqdm(range(total_reps), leave=False)
    for rep_no in r_pbar:
        p_values_list = []
        for m in m_list:
            p_list = get_p_values(m, outputs_tr, outputs_te, total_inner_rep)
            try:
                hm = hmean(p_list)
            except:
                hm = 1.0
            p_values_list.append(hm)
            pbar.set_description(f"{rep_no: 2d} m={m: 3d}: f{hm: 0.5f}")
        r_pbar.set_description(f"rep_no: {rep_no + 1}/{total_reps}")
        p_vals_per_rep_no[rep_no] = p_values_list
    p_values_all_threat_models_dict = p_vals_per_rep_no

    df_list = []
    for name, rep_dict in p_values_all_threat_models_dict.items():
        df = (
            pd.DataFrame(rep_dict)
            .reset_index()
            .assign(m=lambda r: r.index + 2)
            .drop(["index"], axis=1)
        )
        df_list.append(
            pd.melt(df, id_vars=["m"], var_name="rep_no", value_name="p_value").assign(
                threat_model=name
            )
        )
    results_df = pd.concat(df_list)

    results_df.to_hdf(
        "/home/inafen/jupyter_notebooks/dataset_inference/cifar10_{v_type}.h5", v_type
    )
    results_df
else:
    # TODO why is case not used?
    results_df = pd.read_hdf(
        "/home/inafen/jupyter_notebooks/dataset_inference/cifar10_{v_type}.h5", v_type
    )
print(results_df.head())

# remove duplicates from dataframe
results_df_no_duplicates = results_df[~results_df.index.duplicated()]
plt.clf()
ax = sns.lineplot(x="m", y="p_value", data=results_df_no_duplicates)
ax.set_xlabel("Number of samples revealed")
ax.set_ylabel("p-value")
plt.show()
