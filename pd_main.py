import torch
from torchvision.models import vgg16
import models
import os
import numpy as np
import json
import il_config
import dataset
import time
import il_util

def trainer(train_set, test_set, trainloader, testloader, trainloader_inf, testloader_inf, model, optimizer, criterion, device, learning_history_train_dict, learning_history_test_dict, args):
    return
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

    print("----- end training -----")

    """Saving the learned metric dictionary"""
    if arg.save_result:
        if not os.path.exists(arg.result_dir):
            os.makedirs(arg.result_dir)

        if arg.learned_metric == "iteration":
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_iteration_seed{}_train.json".format(arg.dataset
                                   ,arg.model, seed)), "w") as f:
                json.dump(learned_metric_train, f)
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_test_iteration_seed{}_test.json".format(arg.dataset
                                   ,arg.model, seed)), "w") as f:
                json.dump(learned_metric_test, f)
        elif arg.learned_metric == "epoch":
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_train_epoch_seed{}_train.json".format(arg.dataset
                                   ,arg.model, seed)), "w") as f:
                json.dump(learned_metric_train, f)
            with open(os.path.join(arg.result_dir, "{}-{}-learned_metric_test_epoch_seed{}_test.json".format(arg.dataset
                                   ,arg.model, seed)), "w") as f:
                json.dump(learned_metric_test, f)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    arg = config.parse_arguments()
    if not arg.skip_training:
        for seed in arg.seeds:
            print("-------- starting seed {} --------".format(seed))
            main(arg, seed)
    train_avg_score = util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_train.json")
    test_avg_score = util.avg_result(os.path.join(os.getcwd(), arg.result_dir), "_test.json")

    if not os.path.exists(arg.result_dir + "/avg"):
        os.makedirs(arg.result_dir + "/avg")

    if arg.learned_metric == "iteration":
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_iteration_seed{}_train.json".format(arg.dataset
                               ,arg.model, seed)), "w") as f:
            json.dump(train_avg_score, f)
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_iteration_seed{}_test.json".format(arg.dataset
                               ,arg.model, seed)), "w") as f:
            json.dump(test_avg_score, f)
    elif arg.learned_metric == "epoch":
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_train_epoch_seed{}_train.json".format(arg.dataset
                               ,arg.model, seed)), "w") as f:
            json.dump(train_avg_score, f)
        with open(os.path.join(arg.result_dir + "/avg", "{}-{}-learned_metric_test_epoch_seed{}_test.json".format(arg.dataset
                               ,arg.model, seed)), "w") as f:
            json.dump(test_avg_score, f)
    else:
        raise NotImplementedError