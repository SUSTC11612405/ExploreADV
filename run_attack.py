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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='mnist', help='cifar10, mnist, stl10')
    parser.add_argument('--path_model', type=str, default='./models/convSmallRELU__Point.onnx', help='path to the trained model')
    parser.add_argument('--eps', type=float, default=1.0, help='max perturbation size on each pixel')
    parser.add_argument('--region', type=str, default='whole', help='whole, top, bottom, left, right, select')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of pixels allowed to perturb')
    parser.add_argument('--imperceivable', action="store_true", help='whether to have imperceivable perturbation')
    parser.add_argument('--n_examples', type=int, default=100)
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
    masks = {}
    if args.region == 'select':
        from draw import region_selector
        masks['region'] = region_selector(cln_data)
    else:
        masks['region'] = get_region_mask(cln_data.data, args.region)
    if args.imperceivable:
        from region_proposal import get_sigma_mask
        masks['sigma'] = get_sigma_mask(cln_data.data)
    if args.ratio != 1.0:
        from region_proposal import get_captum_mask
        masks['importance'] = get_captum_mask(model, cln_data.data, true_label.data)
    combined_mask = get_combined_mask(masks, args.ratio)

    # run attack
    # run Deepfool
    if args.dataset == 'stl10':
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0,
                                       clip_min=-1., clip_max=1.)
    else:
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    adv = attack_df.perturb(cln_data, true_label, mask=combined_mask)

    pred_df = predict_from_logits(model(adv))
    df_found = ~torch.eq(true_label, pred_df)
    count_df_found = torch.count_nonzero(df_found).item()
    print("Deepfool attack success rate: {:.0%}({}/{})".format(count_df_found / count_correct, count_df_found, count_correct))
    if count_df_found == 0:
        print("Attack success rate: 0.0%(0/{})".format(count_correct))
        exit(0)

    # run BB
    if args.dataset == 'stl10':
        attack_bb = LinfinityBrendelBethgeAttack(model, steps=100, clip_min=-1., clip_max=1.)
    else:
        attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=combined_mask[df_found])
    adv[df_found] = adv_bb

    if args.dataset == 'stl10':
        import torchvision.transforms as transforms
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                       transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                            std=[1., 1., 1.]),
                                       ])
        diff_adv = torch.abs(invTrans(adv) - invTrans(cln_data))
    else:
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

    # calculate metrics
    PerD = PerceptualDistance(args.dataset)
    if args.dataset == 'stl10':
        distance = PerD.cal_perceptual_distances(invTrans(cln_data), invTrans(adv))
    else:
        distance = PerD.cal_perceptual_distances(cln_data, adv)
    PerD.update(distance, adv.size(0))
    PerD.print_metric()

    # resnet18
    # bottom    l0: 1535.11, l2: 11.79, l_inf: 0.08, ssim: 0.93, CIEDE2000: 185.11
    # top       l0: 1534.22, l2: 10.51, l_inf: 0.07, ssim: 0.91, CIEDE2000: 207.04
    # left      l0: 1535.00, l2: 12.18, l_inf: 0.07, ssim: 0.91, CIEDE2000: 205.18
    # right     l0: 1534.06, l2: 8.89, l_inf: 0.07, ssim: 0.93, CIEDE2000: 191.68
    # I         l0: 1534.67, l2: 6.02, l_inf: 0.06, ssim: 0.93, CIEDE2000: 173.57
    # I+V 14/18 l0: 1535.71, l2: 1.93, l_inf: 0.07, ssim: 0.97, CIEDE2000: 105.08
    # I2        l0: 1534.61, l2: 6.60, l_inf: 0.06, ssim: 0.92, CIEDE2000: 177.91

    # 0.1 rand 13/18    l0: 307.77, l2: 13.39, l_inf: 0.21, ssim: 0.85, CIEDE2000: 282.39
    # 0.1 shap 18/18    l0: 307.83, l2: 12.60, l_inf: 0.23, ssim: 0.86, CIEDE2000: 222.90
    # 0.1 sigma 15/18   l0: 307.80, l2: 9.82, l_inf: 0.20, ssim: 0.92, CIEDE2000: 194.58

    # # non-imperceivable
    # start = attack_df.perturb(cln_data, true_label)
    # adv_normal = attack_bb.perturb(cln_data, start)
    # diff_adv_normal = adv_normal - cln_data
    # pred_adv_normal = predict_from_logits(model(adv_normal))
    # epsilon_normal = torch.amax(torch.abs(diff_adv_normal), dim=(1, 2, 3))
    #
    # PerD2 = PerceptualDistance(args.dataset)
    # distance = PerD2.cal_perceptual_distances(cln_data, adv_normal)
    # PerD2.update(distance, adv.size(0))
    # PerD2.print_metric()
    #
    # idx2name = lambda idx: names[idx]
    #
    # plt.figure(figsize=(10, 8))
    # for ii in range(count_found):
    #     # clean image
    #     plt.subplot(3, count_found * 2, 2 * ii + 1)
    #     _imshow(cln_data[ii])
    #     plt.title("clean \n pred: {}".format(idx2name(pred_cln[ii])))
    #     # adv image
    #     plt.subplot(3, count_found * 2, 2 * count_found + 2 * ii + 1)
    #     _imshow(adv_normal[ii])
    #     plt.title("minimal adv \n pred: {}".format(idx2name(pred_adv_normal[ii])))
    #     # adv difference
    #     plt.subplot(3, count_found * 2, 2 * count_found + 2 * ii + 2)
    #     _imshow_diff(diff_adv_normal[ii])
    #     plt.title("Difference \n epsilon: {:.2}".format(epsilon_normal[ii]))
    #     # imperceivable adv image
    #     plt.subplot(3, count_found * 2, 4 * count_found + 2 * ii + 1)
    #     _imshow(adv[ii])
    #     plt.title("imperceivable adv \n pred: {}".format(idx2name(pred_adv[ii])))
    #     # imperceivable adv difference
    #     plt.subplot(3, count_found * 2, 4 * count_found + 2 * ii + 2)
    #     _imshow_diff(diff_adv[ii])
    #     plt.title("Difference \n epsilon: {:.2}".format(epsilon[ii]))
    # plt.tight_layout()
    # plt.show()

    # visualize the results
    visualize = False
    if visualize:
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
            # adv difference
            plt.subplot(3, count_found, 2 * count_found + ii+1)
            _imshow(diff_adv[ii])
            plt.title("Difference \n epsilon: {:.2}".format(epsilon[ii]))
        plt.tight_layout()
        plt.show()