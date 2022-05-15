import torch
import torch.nn as nn
import numpy as np
import argparse

import onnx
from onnx2pytorch import ConvertModel

from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
from eval.eval_metric import PerceptualDistance
from region_proposal import get_region_mask, get_combined_mask
from utils import predict_from_logits, _imshow, _imshow_diff
import matplotlib.pyplot as plt


def load_model(dataset, path):
    if dataset in ['mnist', 'cifar10'] and path.endswith('onnx'):
        # load the ONNX model and convert to Pytorch model
        onnx_model = onnx.load(path)
        pytorch_model = ConvertModel(onnx_model, experimental=True)
    elif dataset == 'stl10':
        from stl10 import stl10
        pytorch_model = stl10(32, pretrained=True)
    elif dataset == 'imagenet':
        from torchvision.models import resnet18, efficientnet_b3, mobilenet_v2
        pytorch_model = mobilenet_v2(pretrained=True)
    else:
        error = "Only onnx models and a stl10 model supported"
        raise NotImplementedError(error)
    model = pytorch_model
    model.to(device)
    model.eval()

    return model


def get_dataloader_with_names(dataset, n_examples):
    if dataset == 'mnist':
        from dataloader import get_mnist_test_loader
        loader = get_mnist_test_loader(batch_size=n_examples)
        names = list(range(10))
    elif dataset == 'cifar10':
        from dataloader import get_cifar10_test_loader
        loader = get_cifar10_test_loader(batch_size=n_examples)
        names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset == 'stl10':
        from dataloader import get_stl10_test_loader
        loader = get_stl10_test_loader(batch_size=n_examples)
        names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset == 'imagenet':
        from dataloader import get_imagenet_val_loader, load_imagenet_class_names
        loader = get_imagenet_val_loader(batch_size=n_examples)
        names = load_imagenet_class_names()
    else:
        error = "Only mnist, cifar10, stl10, imagenet data supported"
        raise NotImplementedError(error)
    return loader, names

def get_epsilon(cln_data, true_label, mask_name, target_layers, n=10):
    from region_proposal import get_nbyn_mask, get_captum_mask, get_gradcam_mask, get_gradcamplusplus_mask
    if mask_name == 'captum':
        mask = get_captum_mask(model, cln_data, true_label, False)
    elif mask_name == 'captum_correction':
        mask = get_captum_mask(model, cln_data, true_label)
    elif mask_name == 'gradcam':
        mask = get_gradcam_mask(model, cln_data, target_layers)
    elif mask_name == 'gradcam++':
        mask = get_gradcamplusplus_mask(model, cln_data, target_layers)
    else:
        error = "Only captum, captum_correction, gradcam, gradcam++ mask supported"
        raise NotImplementedError(error)

    w, h = cln_data.data.shape[-2:]
    score = 0
    pos = (0, 0)

    for i in range(w - n):
        for j in range(h - n):
            if mask_name in ['gradcam', 'gradcam++']:
                mask_sum = np.sum(mask[:, i:i + n, j:j + n])
            else:
                mask_sum = np.sum(mask[:, :, i:i + n, j:j + n])
            if mask_sum > score:
                pos = (i, j)
                score = mask_sum

    mask_n = get_nbyn_mask(cln_data.data, n, pos[0], pos[1])

    attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    adv = attack_df.perturb(cln_data, true_label, mask=mask_n)

    pred_df = predict_from_logits(model(adv))
    df_found = ~torch.eq(true_label, pred_df)
    count_df_found = torch.count_nonzero(df_found).item()
    if count_df_found == 0:
        return 1.0

    # run BB
    attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    adv_bb = attack_bb.perturb(cln_data, adv, mask=mask_n)

    diff = torch.abs(adv_bb - cln_data)
    epsilon = torch.amax(diff, dim=(1, 2, 3))[0].item()
    return epsilon

if __name__ == '__main__':

    # dataset = 'mnist'
    # path_model = './models/mnist_relu_9_200.onnx'
    # path_model = './models/convSmallRELU__Point.onnx'
    dataset = 'cifar10'
    # path_model = "./models/cifar10_relu_6_500.onnx"
    # path_model = "./models/convMedGSIGMOID__Point.onnx"
    path_model = "./models/convBigRELU__DiffAI_cifar10.onnx"
    # path_model = "./models/ResNet18_PGD_cifar10.onnx"
    # dataset = 'stl10'
    n_examples = 1
    eps = 1.0

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(dataset, path_model)
    target_layers = [model.Conv_25]

    masks = ['captum', 'captum_correction', 'gradcam', 'gradcam++']

    # load data
    loader, names = get_dataloader_with_names(dataset, n_examples)
    index = 0
    count = 0

    for cln_data, true_label in loader:
        if count == 100:
            break
        print(index)
        cln_data, true_label = cln_data.to(device), true_label.to(device)

        # calculate raw precision
        pred_cln = predict_from_logits(model(cln_data))
        correct = torch.eq(true_label, pred_cln)
        count_correct = torch.count_nonzero(correct).item()
        if count_correct == 0:
            index += 1
            continue

        cln_data = cln_data[correct]
        true_label = true_label[correct]
        pred_cln = pred_cln[correct]
        epsilons = {}
        for mask_name in masks:
            epsilons[mask_name] = get_epsilon(cln_data, true_label, mask_name, target_layers)

        with open('cifar10_convBigRELU_masks.csv', mode='a') as file:
            file.write("{},{:.4},{:.4},{:.4},{:.4}\n"
                       .format(index, *[epsilons[i] for i in masks]))
        index += 1
        count += 1
