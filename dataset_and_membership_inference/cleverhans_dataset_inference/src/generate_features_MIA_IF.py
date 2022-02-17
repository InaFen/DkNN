from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import time
from cleverhans_dataset_inference.src.attacks import norms_linf_squeezed, norms_l1_squeezed, norms_l2_squeezed, loss_mingd, norms_l2, l1_dir_topk

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


def mingd(model, X, y, distance, target,alpha_l_1 = 1.0, alpha_l_2 = 0.01, alpha_l_inf = 0.001, num_iter = 500, k = 100, gap = 0.001):
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
            remaining[remaining_temp] = new_remaining #TODO changed to remaining_temp
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

def get_mingd_vulnerability_MIA(loader, model, num_images=1000, batch_size = 1000, regressor_embed = 0, num_classes = 10):
    max_iter = num_images / batch_size
    lp_dist = [[], [], []]
    ex_skipped = 0
    for i, batch in enumerate(loader):
        if regressor_embed == 1:  ##We need an extra set of `distinct images for training the confidence regressor
            if (ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j, distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(num_classes):
                #X, y = batch[0].to(device), batch[1].to(device) #TODO needed?
                X,y = batch[0], batch[1]
                # args.lamb = 0.0001
                delta = mingd(model, X, y, distance, target=y * 0 + target_i)
                yp = model(X + delta)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i + 1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim=-1)
    print(full_d.shape)

    return full_d


def feature_extractor_MIA(train_loader, test_loader, student, test_distance_path = "/home/inafen/jupyter_notebooks/dataset_inference/test_distance_vulerability.pt", train_distance_path =  "/home/inafen/jupyter_notebooks/dataset_inference/train_distance_vulerability.pt"):
    """
    Specolized feature extractor for MIA for CIFAR10 dataset

    :param train_loader: Members
    :param test_loader: Non-members
    :param student: pytorch model
    :return:
    """

    student.eval()

    # IF: get the vulnearbility (distance) of the test data and save it
    test_d = get_mingd_vulnerability_MIA(test_loader, student, num_images=1000)
    print(test_d)
    torch.save(test_d, test_distance_path)
    # IF: get the vulnearbility (distance) of the train data and save it
    train_d = get_mingd_vulnerability_MIA(train_loader, student, num_images= 1000)
    print(train_d)
    torch.save(train_d, train_distance_path)



