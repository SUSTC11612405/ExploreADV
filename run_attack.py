import torch
import torch.nn as nn
import numpy as np
import argparse

import onnx
from onnx2pytorch import ConvertModel

from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
from region_proposal import get_region_mask, get_combined_mask
from utils import predict_from_logits, _imshow
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
        from torchvision.models import resnet18, efficientnet_b3
        pytorch_model = efficientnet_b3(pretrained=True)
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


def get_shap_loader(dataset, n_examples=100):
    if dataset == 'mnist':
        from dataloader import get_mnist_train_loader
        loader = get_mnist_train_loader(batch_size=n_examples)
    elif dataset == 'cifar10':
        from dataloader import get_cifar10_train_loader
        loader = get_cifar10_train_loader(batch_size=n_examples)
    elif dataset == 'stl10':
        from dataloader import get_stl10_train_loader
        loader = get_stl10_train_loader(batch_size=n_examples)
    elif dataset == 'imagenet':
        from dataloader import get_imagenet_val_loader, load_imagenet_class_names
        loader = get_imagenet_val_loader(batch_size=n_examples)
    else:
        error = "Only mnist, cifar10, stl10, imagenet data supported"
        raise NotImplementedError(error)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist, stl10')
    parser.add_argument('--path_model', type=str, default='./models/cifar10_2_255.onnx', help='path to the trained model')
    parser.add_argument('--eps', type=float, default=1.0, help='max perturbation size on each pixel')
    parser.add_argument('--region', type=str, default='whole', help='whole, top, bottom, left, right, select')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of pixels allowed to perturb')
    parser.add_argument('--imperceivable', action="store_true", help='whether to have imperceivable perturbation')
    parser.add_argument('--n_examples', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./dataset')

    args = parser.parse_args()

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(args.dataset, args.path_model)

    # load data
    loader, names = get_dataloader_with_names(args.dataset, args.n_examples)

    for cln_data, true_label in loader:
            break
    cln_data, true_label = cln_data.to(device), true_label.to(device)

    # calculate raw precision
    pred_cln = predict_from_logits(model(cln_data))
    correct = torch.eq(true_label, pred_cln)
    count_correct = torch.count_nonzero(correct).item()
    print("Raw model precision: {:.0%}({}/{})".format(count_correct / args.n_examples, count_correct, args.n_examples))
    cln_data = cln_data[correct]
    true_label = true_label[correct]
    pred_cln = pred_cln[correct]

    # generate masks
    masks = []
    if args.region == 'select':
        from draw import region_selector
        masks.append(region_selector(cln_data))
    else:
        masks.append(get_region_mask(cln_data.data, args.region))
    if args.imperceivable:
        from region_proposal import get_sigma_mask
        masks.append(get_sigma_mask(cln_data.data))
    if args.ratio != 1.0:
        from region_proposal import get_shap_mask, get_shap_explainer
        shap_loader = get_shap_loader(args.dataset)
        for background, _ in shap_loader:
            break
        background = background.to(device)
        e = get_shap_explainer(model, background)
        masks.append(get_shap_mask(cln_data.data, e))
    combined_mask = get_combined_mask(masks, args.ratio)

    # run attack
    # run Deepfool
    attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    adv = attack_df.perturb(cln_data, true_label, mask=combined_mask)

    pred_df = predict_from_logits(model(adv))
    df_found = ~torch.eq(true_label, pred_df)
    count_df_found = torch.count_nonzero(df_found).item()
    if count_df_found == 0:
        print("Attack success rate: 0.0%(0/{})".format(count_correct))
        exit(0)

    # run BB
    attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=combined_mask[df_found])
    adv[df_found] = adv_bb

    diff_adv = torch.abs(adv - cln_data)
    epsilon = torch.amax(diff_adv, dim=(1, 2, 3))

    pred_adv = predict_from_logits(model(adv))
    found = torch.logical_and(~torch.eq(true_label, pred_adv), torch.lt(epsilon, args.eps))
    count_found = torch.count_nonzero(found).item()
    print("Attack success rate: {:.0%}({}/{})".format(count_found / count_correct, count_found, count_correct))

    true_label = true_label[found]
    cln_data = cln_data[found]
    adv = adv[found]
    diff_adv = diff_adv[found]
    epsilon = epsilon[found]
    pred_cln = pred_cln[found]
    pred_adv = pred_adv[found]

    # visualize the results
    idx2name = lambda idx: names[idx]

    plt.figure(figsize=(10, 8))
    for ii in range(count_found):
        # clean image
        plt.subplot(3, count_found, ii+1)
        _imshow(cln_data[ii])
        plt.title("clean \n pred: {}".format(idx2name(pred_cln[ii])))
        # adv image
        plt.subplot(3, count_found, count_found + ii+1)
        _imshow(adv[ii])
        plt.title("adversarial \n pred: {}".format(idx2name(pred_adv[ii])))
        # adv image
        plt.subplot(3, count_found, 2 * count_found + ii+1)
        _imshow(diff_adv[ii])
        plt.title("Difference \n epsilon: {:.2}".format(epsilon[ii]))
    plt.tight_layout()
    plt.show()