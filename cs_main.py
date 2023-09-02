import json
import os

import torch
import numpy as np
from torchvision.models import vgg16
from torchvision.transforms import ToTensor
from tqdm import tqdm
from dataset import *
from models import *
import cs_config


def subset_train(seed, device, subset_ratio, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config.dataset == "cifar10":
        num_train_total = 50000
    num_train = int(num_train_total * subset_ratio)

    indices = torch.randperm(num_train_total)

    train_subset_indices = indices[:num_train]  # used in training
    # held-out example, we get c-scores on this
    test_subset_indices = indices[num_train:]

    train_loader = getDataset(config, subset=train_subset_indices)[2]  # we only need the train loader
    test_loader = getDataset(config, subset=test_subset_indices)[2]  # we only need the train loader

    if config.dataset == "cifar10":
        ecd = vgg16().features
        model = VGGPD(ecd, 10)
        model.to(device)
    else:
        raise NotImplementedError

    if config.crit == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)

    training_acc = []

    for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
        for (imgs, labels), idx in train_loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (imgs, labels), idx in train_loader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            training_acc.append(correct / total)
        model.train()

    trainset_mask = torch.zeros(num_train_total, dtype=torch.bool)
    trainset_mask[test_subset_indices] = True

    trainset_correctness = {}

    model.eval()
    with torch.no_grad():
        for idx in test_subset_indices:
            trainset_correctness[idx.item()] = 0
        for (imgs, labels), idx in test_loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    trainset_correctness[idx[i].item()] = 1

    subset_acc = sum(trainset_correctness.values()) / len(trainset_correctness)
    print(f"Subset Accuracy: {subset_acc:.4f}")
    print(f"Training Accuracy: {training_acc[-1]:.4f}")
    return trainset_correctness


def estimate_cscores(config, device):
    n_runs = config.n_runs  # number of runs
    ss_ratio = config.ss_ratio  # subset ratio

    results = []
    for i_run in range(n_runs):
        print(f'Run {i_run + 1}/{n_runs} ----------------------------------')
        # note that the seed is set as config.seed + i_run since we want to have different seeds for different runs
        # so the sunsetting of the training is not the same for different runs
        results.append(subset_train(config.seed + i_run, device, ss_ratio, config))

    train_rep = {}  # number of times each image is predicted in the loop above
    train_correctness_sum = {}  # sum of correctly prediction for each image in the loop above
    for i_run in tqdm(range(n_runs), desc=f'calculate c-scores'):
        for idx in results[i_run]:  # for each image's prediction's correctness
            if idx not in train_rep:
                train_rep[idx] = 0
                train_correctness_sum[idx] = 0
            train_rep[idx] += 1
            if results[i_run][idx]:
                train_correctness_sum[idx] += 1

    cscores = {}

    avg_train_rep_list = []
    for idx in train_rep:
        cscores[idx] = train_correctness_sum[idx] / train_rep[idx]
        avg_train_rep_list.append(train_rep[idx])
    avg_train_rep = sum(avg_train_rep_list) / len(avg_train_rep_list)

    print(f"-----\n the average number of times each image is predicted is {avg_train_rep}\nwith min={min(avg_train_rep_list)} and max={max(avg_train_rep_list)}\n-----")
    return cscores


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if config.save_result:
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

        if not os.path.exists(config.result_dir + "/model_results"):  # this is for saving the partial results from each trained model
            os.makedirs(config.result_dir + "/model_results")

    cscores = estimate_cscores(config, device)

    if config.save_result:
        with open(os.path.join(config.result_dir, "cs_run{}_{}_trainratio{}_train_avg.json".format(config.n_runs, config.dataset, config.ss_ratio)), "w") as f:
            json.dump(cscores, f)


if __name__ == "__main__":
    config = cs_config.parse_arguments()
    main(config)
