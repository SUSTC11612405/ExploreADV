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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
    parser.add_argument('--path_model', type=str, default='./models/cifar10_2_255.onnx', help='path to the onnx model')
    parser.add_argument('--eps', type=float, default=1.0, help='max perturbation size on each pixel')
    parser.add_argument('--region', type=str, default='whole', help='whole, top, bottom, left, right, center')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of pixels allowed to perturb')
    parser.add_argument('--imperceivable', action="store_true", help='whether to have imperceivable perturbation')
    parser.add_argument('--n_examples', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./dataset')

    args = parser.parse_args()

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the ONNX model and convert to Pytorch model
    onnx_model = onnx.load(args.path_model)
    pytorch_model = ConvertModel(onnx_model, experimental=True)
    model = pytorch_model
    model.to(device)
    model.eval()

    # load data
    if args.dataset == 'mnist':
        from dataloader import get_mnist_test_loader
        loader = get_mnist_test_loader(batch_size=args.n_examples)
        names = list(range(10))
    else:
        from dataloader import get_cifar10_test_loader
        loader = get_cifar10_test_loader(batch_size=args.n_examples)
        names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

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
    masks = [get_region_mask(cln_data.data, args.region)]
    if args.imperceivable:
        from region_proposal import get_sigma_mask
        masks.append(get_sigma_mask(cln_data.data))
    combined_mask = get_combined_mask(masks, args.ratio)

    # run attack
    attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    starting_points = attack_df.perturb(cln_data, true_label, mask=combined_mask)
    attack = LinfinityBrendelBethgeAttack(model, steps=100)
    adv = attack.perturb(cln_data, starting_points, mask=combined_mask)

    diff_adv = torch.abs(adv - cln_data)
    epsilon = torch.amax(diff_adv, dim=(1, 2, 3))

    pred_adv = predict_from_logits(model(adv))
    found = torch.logical_and(~torch.eq(true_label, pred_adv), torch.lt(epsilon, args.eps))
    count_found = torch.count_nonzero(found).item()
    print("Attack success rate: {:.0%}({}/{})".format(count_found / count_correct, count_found, count_correct))

    true_label = true_label[found]
    cln_data = cln_data[found]
    adv = adv[found]
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