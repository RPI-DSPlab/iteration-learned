import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch


def getDataset(arg, subset=False):
    """
    :param config: the configuration file
    :return: trainloader, testloader, valloader
    """
    if arg.dataset == "cifar10":
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
        ])
        test_transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                    ])

        class CIFAR10withIndex(CIFAR10):
            """ We override the CIFAR10 class to return the index of the image as well """

            def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
                super(CIFAR10withIndex, self).__init__(root, train, transform, target_transform, download)

            def __getitem__(self, index):
                # to get (img, target), index
                img, target = super(CIFAR10withIndex, self).__getitem__(index)
                return (img, target), index

        trainset = CIFAR10withIndex('./', transform=train_transform, train=True, download=True)
        testset = CIFAR10withIndex('./', transform=test_transform, train=False, download=True)

        train_indices, val_indices = train_test_split(np.arange(arg.num_samples), train_size=arg.train_ratio,
                                                      test_size=(1 - arg.train_ratio))

        if type(subset) == bool and subset == False:
            trainloader = DataLoader(trainset, batch_size=arg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            trainloader2 = DataLoader(trainset, batch_size=arg.batch_size, shuffle=False, num_workers=4,
                                      pin_memory=True)
            testloader = DataLoader(testset, batch_size=arg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            testloader2 = DataLoader(testset, batch_size=arg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            return trainset, testset, trainloader, testloader, trainloader2, testloader2
        else:
            subset_indices = subset  # if subset is not None, it should be a list of indices
            subset_train = Subset(trainset, subset_indices)
            subset_test = Subset(testset, subset_indices)
            trainloader = DataLoader(subset_train, batch_size=arg.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)
            trainloader2 = DataLoader(subset_train, batch_size=arg.batch_size, shuffle=False, num_workers=4,
                                      pin_memory=True)
            testloader = DataLoader(subset_test, batch_size=arg.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)
            testloader2 = DataLoader(subset_test, batch_size=arg.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)
            return trainset, testset, trainloader, testloader, trainloader2, testloader2
