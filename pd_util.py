import numpy as np
import torch
from torch import autocast
import torch.nn.functional as F


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t, rm_top1=True, dist='l2'):
    """
    knn prediction
    :param feature: feature vector of the current evaluating batch (dim = [B, F]
    :param feature_bank: feature bank of the support set (dim = [F, K]
    :param feature_labels: labels of the support set (dim = [K]
    :param classes: number of classes
    :param knn_k: number of nearest neighbors
    :param knn_t: temperature
    :param rm_top1: whether to remove the nearest pt of current evaluating pt in the train split (explain: this is because
                    the feature vector of the current evaluating pt may also be in the feature bank)
    :param dist: distance metric
    :return: prediction scores for each class (dim = [B, classes]
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    feature_bank = feature_bank.t()  # [F, K].t() -> [K, F]
    B, F = feature.shape  # dim of feature vector of the current evaluating pt
    K, F = feature_bank.shape  # dim feature bank
    """
    B: batch size (ie: 200)
    F: feature dimension (ie: 65536)
    K: number of pts in the feature bank (ie 5000)
    """


    if dist == 'l2':
        knn_dist = 2

    distances = torch.cdist(feature, feature_bank, p=knn_dist)

    # Find the k nearest neighbors of the input feature.
    nearest_neighbors = distances.argsort(dim=1)[:, :knn_k]

    # If `rm_top1` is True, remove the nearest neighbor of the current evaluating point from the list of nearest neighbors.
    if rm_top1:
        mask = torch.ones(nearest_neighbors.shape[1], dtype=torch.bool)
        mask[0] = False  # mask the first element
        nearest_neighbors_dropped = nearest_neighbors[:, mask]
        nearest_labels = feature_labels[nearest_neighbors_dropped]
    else:
        nearest_labels = feature_labels[nearest_neighbors]

    # Compute the weighted scores using the inverse distances
    inv_distances = (1.0 / distances[:, :knn_k])
    knn_scores = torch.zeros(B, classes, device=feature.device)

    for i in range(B):
        for j in range(knn_k - 1 if rm_top1 else knn_k):
            knn_scores[i, nearest_labels[i, j]] += inv_distances[i, j]

    # Apply temperature scaling
    knn_scores /= knn_t

    return knn_scores


def _get_feature_bank_from_kth_layer(model, dataloader, k, args):
    """
    Get feature bank from kth layer of the model
    :param model: the model
    :param dataloader: the dataloader
    :param k: the kth layer
    :return: the feature bank (k-th layer feature for each datapoint) and
            the all label bank (ground truth label for each datapoint)
    """
    # NOTE: dataloader now has the return format of '(img, target), index'
    print(k, 'layer feature bank gotten')
    with torch.no_grad():
        for (img, all_label), idx in dataloader:
            img = img.cuda(non_blocking=True)  # an image from the dataset
            all_label = all_label.cuda(non_blocking=True)

            # the return of model():'None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)'
            if args.half:
                with autocast():
                    _, fms = model(img, k, train=False)
            else:
                _, fms = model(img, k, train=False)
    # print("return value from _get_feature_bank_from_kth_layer:\n", "fms:\n", fms, "\nlen of fms: ", len(fms), "\nall_label\n:", all_label, "\nlen of all_label: ", len(all_label))

    return fms, all_label  # somehow, the shape of fms is (number of image) * (it's feature map size)


def get_knn_prds_k_layer(model, evaloader, floader, k, args, train_split=True):
    """
    Get the knn predictions for the kth layer
    :param model: the model
    :param evaloader: the evaluation dataloader (training or validation)
    :param floader: the feature dataloader (support set)
    :param k: the kth layer
    """
    knn_labels_all = []
    knn_conf_gt_all = []  # This statistics can be noisy
    indices_all = []
    f_bank, all_labels = _get_feature_bank_from_kth_layer(model, floader,
                                                          k)  # get the feature bank and all labels for the support set
    f_bank = f_bank.t().contiguous()
    with torch.no_grad():
        for j, ((imgs, labels), idx) in enumerate(evaloader):
            imgs = imgs.cuda(non_blocking=True)
            labels_b = labels.cuda(non_blocking=True)
            nm_cls = args.num_classes
            if args.half:
                with autocast():
                    _, inp_f_curr = model(imgs, k, train=False)
            else:
                _, inp_f_curr = model(imgs, k, train=False)
            """
            Explanation of the following function:
            knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=args.knn_k, knn_t=1, rm_top1=train_split)
            inp_f_curr is the feature of the image (batch of images) we want to predict it's label
            f_bank is the feature bank of the support set, and we know its ground truth label given all_labels
            We want to use information from the support set (f_bank) to predict the label of the image (inp_f_curr)
            """
            knn_scores = knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=args.knn_k, knn_t=1,
                                     rm_top1=train_split)  # B x C
            knn_probs = F.normalize(knn_scores, p=1, dim=1)
            knn_labels_prd = knn_probs.argmax(1)
            knn_conf_gt = knn_probs.gather(dim=1, index=labels_b[:, None])  # B x 1
            knn_labels_all.append(knn_labels_prd)
            knn_conf_gt_all.append(knn_conf_gt)
            indices_all.append(idx)
        knn_labels_all = torch.cat(knn_labels_all, dim=0)  # N x 1
        knn_conf_gt_all = torch.cat(knn_conf_gt_all, dim=0).squeeze()
        indices_all = np.concatenate(indices_all, 0)
    return knn_labels_all, knn_conf_gt_all, indices_all


def get_prediction_depth(knn_labels_all, max_prediction_depth):
    """
    get prediction depth for a sample. reverse knn labels list and increase the counter until the label is different
    :param knn_labels_all:
    :return:
    """
    pd = 0
    knn_labels_all = list(reversed(knn_labels_all))
    while knn_labels_all[pd] == knn_labels_all[0] and pd <= max_prediction_depth - 2:
        pd += 1
    return max_prediction_depth - pd
