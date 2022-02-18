"""
Functions based on https://github.com/cleverhans-lab/dataset-inference/blob/main/src/

Functions are specialized for MIA use.
Functions which are needed for feature extraction --> feature extraction specialized for MIA --> use minGD to get prediction margin (=distance from decision boundary) of data points
"""

from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import time
from cleverhans_dataset_inference.src.attacks import norms_linf_squeezed, norms_l1_squeezed, norms_l2_squeezed, loss_mingd, norms_l2, l1_dir_topk
import sys
#sys.path.append("./model_src/")
from models_torch import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets.folder import pil_loader

'''Threat Models'''


# A) complete model theft
# --> A.1 Datafree distillation / Zero shot learning
# --> A.2 Fine tuning (on unlabeled data to slightly change decision surface)
# B) Extraction over an API:
# --> B.1 Model extraction using unlabeled data and victim labels
# --> B.2 Model extraction using unlabeled data and victim confidence
# C) Complete data theft:
# --> C.1 Data distillation
# --> C.2 Different architecture/learning rate/optimizer/training epochs
# --> C.3 Coresets
# D) Train a teacher model on a separate dataset (test set)

class PseudoDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len

def get_dataloaders_MIA(dataset, batch_size, pseudo_labels=False, normalize=False, train_shuffle=True, concat=False,
                    concat_factor=1.0):
    if dataset in ["CIFAR10", "CIFAR100"]:
        data_source = datasets.CIFAR10 if dataset == "CIFAR10" else datasets.CIFAR100
        tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2471, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), tr_normalize,
                                              transforms.Lambda(lambda x: x.float())])

        transform_test = transforms.Compose(
            [transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])
        if not train_shuffle:
            print("No Transform")
            transform_train = transform_test
        cifar_train = data_source("../../data", train=True, download=True, transform=transform_train)
        cifar_test = data_source("../../data", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

        if pseudo_labels:
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])

            import pickle, os
            aux_data_filename = "ti_500K_pseudo_labeled.pickle"
            aux_path = os.path.join("../data", aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']

            cifar_train = PseudoDataset(aux_data, aux_targets, transform=transform_train)
            train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=train_shuffle)

    if dataset == "MNIST":
        tr_normalize = transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([transforms.ToTensor(), tr_normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize])  # Change
        mnist_train = datasets.MNIST("../../data", train=True, download=True, transform=transform_train)
        mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


    elif dataset != "AFAD":
        func = {"SVHN": datasets.SVHN, "CIFAR10": datasets.CIFAR10, "CIFAR100": datasets.CIFAR100,
                "MNIST": datasets.MNIST, "ImageNet": datasets.ImageNet}
        norm_mean = {"SVHN": (0.438, 0.444, 0.473), "CIFAR10": (0.4914, 0.4822, 0.4465),
                     "CIFAR100": (0.4914, 0.4822, 0.4465), "MNIST": (0.1307,), "ImageNet": (0.485, 0.456, 0.406)}
        norm_std = {"SVHN": (0.198, 0.201, 0.197), "CIFAR10": (0.2023, 0.1994, 0.2010),
                    "CIFAR100": (0.2023, 0.1994, 0.2010), "MNIST": (0.3081,), "ImageNet": (0.229, 0.224, 0.225)}

        tr_normalize = transforms.Normalize(norm_mean[dataset], norm_std[dataset]) if normalize else transforms.Lambda(
            lambda x: x)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), tr_normalize,
            transforms.Lambda(lambda x: x.float())])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])

        data_source = func[dataset]
        if not train_shuffle:
            print("No Transform")
            transform_train = transform_test

        if dataset == "ImageNet":
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            tr_normalize])
            train_path = '/scratch/ssd001/datasets/imagenet/train'
            test_path = '/scratch/ssd001/datasets/imagenet/val'

            d_train = torchvision.datasets.ImageFolder(train_path, transform=transform)
            d_test = torchvision.datasets.ImageFolder(test_path, transform=transform)
            # d_test = data_source("/scratch/ssd001/datasets/imagenet", split='val', download=False, transform=transform_test)
            # d_train = data_source("/scratch/ssd001/datasets/imagenet", split='train', download=False, transform=transform_train)
        else:
            try:
                d_train = data_source("../data", train=True, download=True, transform=transform_train)
                d_test = data_source("../data", train=False, download=True, transform=transform_test)
            except:
                if concat:
                    d_train = data_source("../data", split='train', download=True, transform=transform_train)
                    d_extra = data_source("../data", split='extra', download=True, transform=transform_test)
                    train_len = d_train.data.shape[0]
                    new_len = int(train_len * concat_factor)
                    d_train.data = d_train.data[:new_len]
                    d_train.labels = d_train.labels[:new_len]
                    d_extra.data = d_extra.data[:50000]
                    d_extra.labels = d_extra.labels[:50000]
                    d_train = torch.utils.data.ConcatDataset([d_train, d_extra])

                else:
                    d_train = data_source("../data", split='train' if not pseudo_labels else 'extra', download=True,
                                          transform=transform_train)
                    d_train.data = d_train.data[:50000]
                    d_train.labels = d_train.labels[:50000]

                d_test = data_source("../data", split='test', download=True, transform=transform_test)

        train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)
        if pseudo_labels and dataset != "SVHN":
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])
            import pickle, os
            aux_data_filename = "ti_500K_pseudo_labeled.pickle"
            aux_path = os.path.join("../data", aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']

            cifar_train = PseudoDataset(aux_data, aux_targets, transform=transform_train)
            train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=train_shuffle)


    return train_loader, test_loader


def mingd_MIA(model, X, y, distance, target,alpha_l_1 = 1.0, alpha_l_2 = 0.01, alpha_l_inf = 0.001, num_iter = 500, k = 100, gap = 0.001):
    start = time.time()
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    criterion = loss_mingd
    norm_map = {"l1": norms_l1_squeezed, "l2": norms_l2_squeezed, "linf": norms_linf_squeezed}
    alpha_map = {"l1": alpha_l_1 / k, "l2": alpha_l_2, "linf": alpha_l_inf}
    alpha = float(alpha_map[distance])

    delta = torch.zeros_like(X, requires_grad=False)
    loss = 0
    for t in range(num_iter):
        if t > 0:
            preds = model(X_r + delta_r)
            new_remaining = (preds.max(1)[1] != target[remaining])
            remaining_temp = remaining.clone()
            remaining[remaining_temp] = new_remaining #IF: changed to remaining_temp to avoid error
        else:
            preds = model(X + delta)
            remaining = (preds.max(1)[1] != target)

        if remaining.sum() == 0: break

        X_r = X[remaining];
        delta_r = delta[remaining]
        delta_r.requires_grad = True
        preds = model(X_r + delta_r)
        loss = -1 * loss_mingd(preds, target[remaining])
        # print(t, loss, remaining.sum().item())
        loss.backward()
        grads = delta_r.grad.detach()
        if distance == "linf":
            delta_r.data += alpha * grads.sign()
        elif distance == "l2":
            delta_r.data += alpha * (grads / norms_l2(grads + 1e-12))
        elif distance == "l1":
            delta_r.data += alpha * l1_dir_topk(grads, delta_r.data, X_r, gap, k)
        delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)  # clip X+delta_r[remaining] to [0,1]
        delta_r.grad.zero_()
        delta[remaining] = delta_r.detach()

    print(
        f"Number of steps = {t + 1} | Failed to convert = {(model(X + delta).max(1)[1] != target).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta


def mingd_unoptimized_MIA(model, X, y, distance,target, alpha_l_1 = 1.0, alpha_l_2 = 0.01, alpha_l_inf = 0.001, num_iter = 500, k = 100, gap = 0.001):
    # Can try \delta = (1/2)* { tanh(w) + 1} — x for box constraints
    start = time.time()
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    # args = vars(args) if type(args) != type({"a":1}) else args
    criterion = loss_mingd
    norm_map = {"l1": norms_l1_squeezed, "l2": norms_l2_squeezed, "linf": norms_linf_squeezed}
    alpha_map = {"l1": alpha_l_1 / k, "l2": alpha_l_2, "linf": alpha_l_inf}
    alpha = float(alpha_map[distance])

    delta = torch.zeros_like(X, requires_grad=True)
    loss = 0
    # ipdb.set_trace()
    for t in range(num_iter):
        preds = model(X + delta)
        remaining = (preds.max(1)[1] != target)
        if remaining.sum() == 0: break
        # loss1 = -1* norm_map[args.distance](delta).mean()
        # loss = -1* loss_mingd(preds[remaining], target[remaining])
        loss = -1 * loss_mingd(preds, target)
        # loss = args.lamb * loss1 + loss2
        # print(t, loss, remaining.sum().item())
        loss.backward()
        grads = delta.grad.detach()
        if distance == "linf":
            delta.data += alpha * grads.sign()
        elif distance == "l2":
            delta.data += alpha * (grads / norms_l2(grads + 1e-12))
        elif distance == "l1":
            delta.data += alpha * l1_dir_topk(grads, delta.data, X, gap, k)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta[remaining] to [0,1]
        delta.grad.zero_()

    print(
        f"Number of steps = {t} | Failed to convert = {(preds.max(1)[1] != target).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta

def get_mingd_vulnerability_MIA(loader, model, num_images=1000, batch_size = 100, regressor_embed = 0, num_classes = 10, device = 'cpu'):
    max_iter = num_images / batch_size
    #IF: three sublists for one distance type (linf, l2, l1) each
    lp_dist = [[], [], []]
    ex_skipped = 0
    #IF: for each image
    for i, batch in enumerate(loader):
        if regressor_embed == 1:  ##We need an extra set of `distinct images for training the confidence regressor
            if (ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        #IF: for each distance
        for j, distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            #IF: for each class
            for target_i in range(num_classes):
                X, y = batch[0].to(device), batch[1].to(device)
                # args.lamb = 0.0001
                delta = mingd_MIA(model, X, y, distance, target=y * 0 + target_i)
                yp = model(X + delta)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                #IF: append all distances for each class
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim=1)
            #IF: append distances for each class (temp_list) for one distance (linf, l1, l2)
            lp_dist[j].append(temp_dist)
        if i + 1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    #IF: for each distance type (linf, l1, l2) have for each element --> for each class --> distance (range 3 for distance types)
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    print(len(lp_d))
    print(lp_d[0].shape)
    #IF: for each image --> for each class --> three distances (linf, l1, l2)
    full_d = torch.cat(lp_d, dim=-1)
    print(full_d.shape)

    return full_d

def feature_extractor_MIA( device: str = 'cpu', test_distance_path: str = "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt", train_distance_path: str =  "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"):
    """
    Specialized feature extractor for MIA for CIFAR10 dataset

    :param device: Which device is used. 'cpu' if CPU is used.
    :param test_distance_path: Path where distances from test data should be saved
    :param train_distance_path: Path where distances from train data should be saved
    """
    train_loader, test_loader = get_dataloaders_MIA(dataset = "CIFAR10", batch_size = 100, pseudo_labels=False, train_shuffle=False)

    #IF: get model
    student = Net()
    try:
        student = student.to(device)
        student.load_state_dict(torch.load("/home/inafen/jupyter_notebooks/dataset_inference/model_torch.pth", map_location=device))
    except:
        student = nn.DataParallel(student).to(device)
        student.load_state_dict(torch.load("/home/inafen/jupyter_notebooks/dataset_inference/model_torch.pth", map_location=device))

    # IF: get the vulnearbility (distance) of the test data and save it
    test_d = get_mingd_vulnerability_MIA(test_loader, student, num_images=1000)
    print(test_d)
    torch.save(test_d, test_distance_path)
    # IF: get the vulnearbility (distance) of the train data and save it
    train_d = get_mingd_vulnerability_MIA(train_loader, student, num_images= 1000)
    print(train_d)
    torch.save(train_d, train_distance_path)



