import torch
from torchvision.models import vgg16
import models
import os
import numpy as np
import json
import config
import dataset
import time
import util

def trainer(train_set, test_set, trainloader, testloader, trainloader_inf, testloader_inf, model, optimizer, criterion, device, learning_history_train_dict, learning_history_test_dict, args):
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
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()
            curr_iteration += 1
            if args.learned_metric == "iteration":  # if we are using iteration learned metric
                model.eval()
                for (imgs_inf, labels_inf), idx_inf in trainloader_inf:
                    imgs_inf, labels_inf = imgs_inf.to(device), labels_inf.to(device)
                    outputs_inf = model(imgs_inf)
                    _, predicted_inf = torch.max(outputs_inf.data, 1)
                    for i in range(len(idx_inf)):
                        learning_history_train_dict[idx_inf[i].item()].append(labels_inf[i].item() == predicted_inf[i].item())
                for (imgs_inf, labels_inf), idx_inf in testloader_inf:
                    imgs_inf, labels_inf = imgs_inf.to(device), labels_inf.to(device)
                    outputs_inf = model(imgs_inf)
                    _, predicted_inf = torch.max(outputs_inf.data, 1)
                    for i in range(len(idx_inf)):
                        learning_history_test_dict[idx_inf[i].item()].append(labels_inf[i].item() == predicted_inf[i].item())
                model.train()

        cos_scheduler.step()
        train_acc /= len(trainloader.dataset)
        train_loss /= len(trainloader.dataset)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        end_time_train = time.time()

        if args.learned_metric == "epoch":  # if we are using epoch learned metric
            model.eval()
            for (imgs_inf, labels_inf), idx_inf in trainloader_inf:
                imgs_inf, labels_inf = imgs_inf.to(device), labels_inf.to(device)
                outputs_inf = model(imgs_inf)
                _, predicted_inf = torch.max(outputs_inf.data, 1)
                for i in range(len(idx_inf)):
                    learning_history_train_dict[idx_inf[i].item()].append(labels_inf[i].item() == predicted_inf[i].item())
            for (imgs_inf, labels_inf), idx_inf in testloader_inf:
                imgs_inf, labels_inf = imgs_inf.to(device), labels_inf.to(device)
                outputs_inf = model(imgs_inf)
                _, predicted_inf = torch.max(outputs_inf.data, 1)
                for i in range(len(idx_inf)):
                    learning_history_test_dict[idx_inf[i].item()].append(labels_inf[i].item() == predicted_inf[i].item())
            model.train()
        if curr_iteration > args.iterations:
            break
        end_time_after_inference = time.time()

        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}, training time: {:.2f}, inference time: '
              '{:.6f}'.format(epoch, train_loss, train_acc, end_time_train - start_time_train,
                              end_time_after_inference - end_time_train))

def determineLearnedMetric(learning_history_dict):
    """
    This function determines the learned metric for the model, it finds the iteration or epoch which the model has
    learned a datapoint
    :param learning_history_dict: dictionary containing the learning history of the model, if -1, then the model can't learn
        this datapoint
    """
    learned_metric_dict = {}
    for i in learning_history_dict:
        learned_itr = 0
        learned_bool = False
        curr_itr = 0
        for j in learning_history_dict[i]:
            if j == True:  # if the model has learned this datapoint
                if learned_bool == False:
                    learned_itr = curr_itr
                learned_bool = True
            else:  # if the model has not learned this datapoint or has forgotten it
                learned_bool = False
            curr_itr += 1

        if learned_bool == True:
            learned_metric_dict[i] = learned_itr
        else:
            learned_metric_dict[i] = -1

    return learned_metric_dict

def main(arg, seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # loading the dataset, note that trainloader_inf and testloader_inf are for inference for the learned metric
    train_set, test_set, trainloader, testloader, trainloader_inf, testloader_inf, val_split = dataset.get_dataset(arg)

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

    """Learning History Dictionary is going to store the learning history of the model, at each iteration or epoch,
    a data point is either correctly classified or misclassified. This directory will be further used to determine the 
    iteration learned or epoch learned metric"""
    learning_history_train_dict = {}
    for _, idx in trainloader_inf:
        for i in idx:
            learning_history_train_dict[i.item()] = list()
    learning_history_test_dict = {}
    for _, idx in testloader_inf:
        for i in idx:
            learning_history_test_dict[i.item()] = list()


    print("----- start training -----")
    trainer(train_set, test_set, trainloader, testloader, trainloader_inf, testloader_inf, model, optimizer, criterion, device, learning_history_train_dict, learning_history_test_dict, arg)
    learned_metric_train = determineLearnedMetric(learning_history_train_dict)
    learned_metric_test = determineLearnedMetric(learning_history_test_dict)

    print("----- start training -----")

    """Saving the learned metric dictionary"""
    if arg.save_result:
        if not os.path.exists(arg.result_dir):
            os.makedirs(arg.result_dir)

        # remove previous results
        files = os.listdir(arg.result_dir)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(arg.result_dir, file)
                os.remove(file_path)

        if arg.learned_metric == "iteration":
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_iteration_seed{}_train.json".format(arg.dataset
                                   ,arg.model, seed)), "w") as f:
                json.dump(learned_metric_train, f)
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_test_iteration_seed{}_test.json".format(arg.dataset
                                   ,arg.model), seed), "w") as f:
                json.dump(learned_metric_test, f)
        elif arg.learned_metric == "epoch":
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_train_epoch_seed{}_train.json".format(arg.dataset
                                   ,arg.model), seed), "w") as f:
                json.dump(learned_metric_train, f)
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_test_epoch_seed{}_test.json".format(arg.dataset
                                   ,arg.model), seed), "w") as f:
                json.dump(learned_metric_test, f)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    arg = config.parse_arguments()
    if not arg.skip_training:
        for seed in arg.seeds:
            main(arg, seed)
    _, train_avg_score = util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_train.json")
    _, test_avg_score = util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_test.json")

    if not os.path.exists(arg.result_dir + "/avg"):
        os.makedirs(arg.result_dir + "/avg")

    if arg.learned_metric == "iteration":
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_iteration_seed{}_train.json".format(arg.dataset
                               ,arg.model, seed)), "w") as f:
            json.dump(train_avg_score, f)
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_iteration_seed{}_test.json".format(arg.dataset
                               ,arg.model), seed), "w") as f:
            json.dump(test_avg_score, f)
    elif arg.learned_metric == "epoch":
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_train_epoch_seed{}_train.json".format(arg.dataset
                               ,arg.model), seed), "w") as f:
            json.dump(train_avg_score, f)
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_test_epoch_seed{}_test.json".format(arg.dataset
                               ,arg.model), seed), "w") as f:
            json.dump(test_avg_score, f)
    else:
        raise NotImplementedError