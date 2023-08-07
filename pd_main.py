import collections
import torch
from torchvision.models import vgg16
import models
import os
import numpy as np
import json
import pd_config
import dataset
import time
import pd_util
from tqdm import tqdm

def trainer(trainloader, testloader, model, optimizer, criterion, device, args):
    curr_iteration = 0
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    print('------ Training started on {} with total number of {} epochs ------'.format(device, args.num_epochs))
    for epoch in range(args.num_epochs):
        # time each epoch
        start_time_train = time.time()
        train_acc = 0
        train_loss = 0
        for (imgs, labels), idx in trainloader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()
            curr_iteration += 1
        cos_scheduler.step()
        train_acc /= len(trainloader.dataset)
        train_loss /= len(trainloader.dataset)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        end_time_train = time.time()
        with torch.no_grad():
            test_acc = 0
            test_loss = 0
            for (imgs, labels), idx in testloader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()
                test_loss += loss.item()
            test_acc /= len(testloader.dataset)
            test_loss /= len(testloader.dataset)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

        if curr_iteration > args.iterations:
            break
        end_time_after_inference = time.time()
        if epoch % 20 == 0:
            print(
                'Epoch: {}, Training Loss: {:.6f}, Test Loss: {:.6f}, Training Accuracy: {:.2f}, Test Accuracy: {:.2f}, training time: {:.2f}, inference time: '
                '{:.6f}'.format(epoch, train_loss, test_loss, train_acc, test_acc, end_time_train - start_time_train,
                                end_time_after_inference - end_time_train))
    return model, history


def main(arg, seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    """trainloader and testloader are the dataloaders for the training and testing sets respectively
    trainloader2 and testloader2 are the dataloaders for predicting the depth"""
    _, _, trainloader, testloader, trainloader2, testloader2 = dataset.getDataset(arg)

    if arg.dataset == "cifar10":
        ecd = vgg16().features
        model = models.VGGPD(ecd, 10)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model.to(device)

    if arg.crit == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    optimizer = torch.optim.SGD(model.parameters(), lr=arg.learning_rate, momentum=0.9, weight_decay=5e-4)

    if not arg.resume:
        if not os.path.exists(arg.model_dir):
            os.makedirs(arg.model_dir)
        print("----- start training -----")
        model, _ = trainer(trainloader, testloader, model, optimizer, criterion, device, arg)
        # save the model
        torch.save(model.state_dict(), os.path.join(arg.model_dir, 'ms{}_{}sgd{}.pt'.format(arg.model, arg.dataset, seed)))
        print("----- end training -----")

    else:
        print('loading model from ckpt...')
        model.load_state_dict(torch.load(
            os.path.join(arg.model_dir, 'ms{}_{}sgd{}.pt'.format(arg.model, arg.dataset, seed))))

    index_knn_y_train = collections.defaultdict(list)
    index_pd_train = collections.defaultdict(int)
    knn_gt_conf_all_train = collections.defaultdict(list)
    index_knn_y_test = collections.defaultdict(list)
    index_pd_test = collections.defaultdict(int)
    knn_gt_conf_all_test = collections.defaultdict(list)

    # ------------------ training set pd ------------------
    if not os.path.exists(os.path.join(arg.result_dir)):
        os.makedirs(os.path.join(arg.result_dir))
    print("----- start obtaining training set pd -----")
    for k in tqdm(range(model.get_num_layers())):
        knn_labels, knn_conf_gt_all, indices_all = pd_util.get_knn_prds_k_layer(model, trainloader, trainloader2,
                                                                        k, arg, True)
        for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
            index_knn_y_train[int(idx)].append(knn_l.item())
            knn_gt_conf_all_train[int(idx)].append(knn_conf_gt.item())
    for idx, knn_ls in index_knn_y_train.items():
        index_pd_train[idx] = (pd_util.get_prediction_depth(knn_ls, model.get_num_layers()))
    with open(os.path.join(arg.result_dir, 'ms{}train_seed{}_{}_trainpd.json'.format(arg.model, seed, arg.dataset)),
              'w') as f:
        json.dump(index_pd_train, f)

    # ------------------ testing set pd ------------------
    print("----- start obtaining testing set pd -----")
    for k in tqdm(range(model.get_num_layers())):
        knn_labels, knn_conf_gt_all, indices_all = pd_util.get_knn_prds_k_layer(model, testloader, testloader2,
                                                                        k, arg, True)
        for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
            index_knn_y_test[int(idx)].append(knn_l.item())
            knn_gt_conf_all_test[int(idx)].append(knn_conf_gt.item())
    for idx, knn_ls in index_knn_y_test.items():
        index_pd_test[idx] = (pd_util.get_prediction_depth(knn_ls, model.get_num_layers()))
    with open(os.path.join(arg.result_dir, 'ms{}train_seed{}_{}_testpd.json'.format(arg.model, seed, arg.dataset)),
                'w') as f:
            json.dump(index_pd_test, f)


if __name__ == '__main__':
    arg = pd_config.parse_arguments()
    for seed in arg.seeds:
        print("-------- starting seed {} --------".format(seed))
        main(arg, seed)
    train_avg_score = pd_util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_trainpd.json")
    test_avg_score = pd_util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_testpd.json")

    if not os.path.exists(arg.result_dir + "/avg"):
        os.makedirs(arg.result_dir + "/avg")

    with open(os.path.join(arg.result_dir, 'avg', '{}-{}-pd_metric_train_avg.json'.format(arg.dataset, arg.model)), 'w') as f:
        json.dump(train_avg_score, f)
    with open(os.path.join(arg.result_dir, 'avg', '{}-{}-pd_metric_test_avg.json'.format(arg.dataset, arg.model)), 'w') as f:
        json.dump(test_avg_score, f)