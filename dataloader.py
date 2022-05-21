import os
import pathlib
from torch.utils.data.dataset import Subset

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


ROOT_PATH = os.path.expanduser(".")
DATA_PATH = os.path.join(ROOT_PATH, "dataset")


def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def get_mnist_train_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_train"
    return loader


def get_mnist_test_loader(batch_size, get_size=False, shuffle=False):
    dataset = datasets.MNIST(DATA_PATH, train=False, download=True,
                             transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_test"
    if get_size:
        return loader, len(dataset)
    return loader


def get_cifar10_train_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_train"
    return loader


def get_cifar10_test_loader(batch_size, get_size=False, shuffle=False):
    dataset = datasets.CIFAR10(DATA_PATH, train=False, download=True,
                               transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_test"
    if get_size:
        return loader, len(dataset)
    return loader


def get_stl10_train_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.STL10(DATA_PATH, split='train', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "stl10_train"
    return loader


def get_stl10_test_loader(batch_size, get_size=False, shuffle=False):
    dataset = datasets.STL10(DATA_PATH, split='test', download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loader.name = "stl10_test"
    if get_size:
        return loader, len(dataset)
    return loader


def get_imagenet_val_loader(batch_size, shuffle=False):
    path = os.path.join(DATA_PATH, 'imagenet/val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "imagenet_val"
    return loader


def load_imagenet_class_names():
    import json
    path = os.path.join(DATA_PATH, 'imagenet/imagenet_class_index.json')
    with open(path) as f:
        class_names = json.load(f)
    names = [class_names[str(i)][1] for i in range(len(class_names))]
    return names
