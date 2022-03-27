import os
import pathlib
from torch.utils.data.dataset import Subset

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


ROOT_PATH = os.path.expanduser("~/.advertorch")
DATA_PATH = os.path.join(ROOT_PATH, "data")


def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def get_mnist_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_train"
    return loader


def get_mnist_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_test"
    return loader


def get_cifar10_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_train"
    return loader


def get_cifar10_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=False, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_test"
    return loader