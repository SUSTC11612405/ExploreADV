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
import torchvision.transforms as transforms


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
    parser.add_argument('--dataset', type=str, default='stl10', help='cifar10, mnist, stl10')
    parser.add_argument('--path_model', type=str, default='./models/cifar10_ResNet18_PGD.onnx', help='path to the trained model')
    parser.add_argument('--eps', type=float, default=1.0, help='max perturbation size on each pixel')
    parser.add_argument('--region', type=str, default='select', help='whole, top, bottom, left, right, select')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of pixels allowed to perturb')
    parser.add_argument('--imperceivable', action="store_true", help='whether to have imperceivable perturbation')
    parser.add_argument('--n_examples', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./dataset')

    args = parser.parse_args()

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.dataset == 'stl10':
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                       transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                            std=[1., 1., 1.]),
                                       ])
    elif args.dataset == 'imagenet':
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                            std=[1., 1., 1.]),
                                       ])

    # load model
    model = load_model(args.dataset, args.path_model)

    # load data
    loader, names = get_dataloader_with_names(args.dataset, args.n_examples)

    count = 0
    target = 1501
    # target = 1430
    # target = 34
    for cln_data, true_label in loader:
        if count == target:
            break
        count += 1
    cln_data, true_label = cln_data.to(device), true_label.to(device)

    # calculate raw precision
    pred_cln = predict_from_logits(model(cln_data))
    correct = torch.eq(true_label, pred_cln)
    count_correct = torch.count_nonzero(correct).item()
    print("Raw model precision: {:.0%}({}/{})".format(count_correct / args.n_examples, count_correct, args.n_examples))
    cln_data = cln_data[correct]
    true_label = true_label[correct]
    pred_cln = pred_cln[correct]

    from draw import region_selector
    from region_proposal import get_sigma_mask

    if args.dataset in ('stl10', 'imagenet'):
        regional_mask = region_selector(invTrans(cln_data))
    else:
        regional_mask = region_selector(cln_data)
    sigma_mask = get_sigma_mask(cln_data.data)

    mask_r = torch.tensor(regional_mask, dtype=torch.float32)
    mask_i = torch.tensor(regional_mask * sigma_mask, dtype=torch.float32)

    # run attack
    # run Deepfool
    if args.dataset == 'stl10':
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0,
                                       clip_min=-1., clip_max=1., nb_iter=200, loosen_rate=1.1, loosen_num=20)
    elif args.dataset == 'imagenet':
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=4.0,
                                       clip_min=-1.8, clip_max=2.2, nb_iter=200, loosen_rate=1.1, loosen_num=20)
    else:
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    adv_r = attack_df.perturb(cln_data, true_label, mask=mask_r)
    adv_i = attack_df.perturb(cln_data, true_label, mask=mask_i)

    pred_df_r = predict_from_logits(model(adv_r))
    pred_df_i = predict_from_logits(model(adv_i))
    df_found_r = ~torch.eq(true_label, pred_df_r)
    df_found_i = ~torch.eq(true_label, pred_df_i)
    count_df_found_r = torch.count_nonzero(df_found_r).item()
    count_df_found_i = torch.count_nonzero(df_found_i).item()
    print(count_df_found_r, count_df_found_i)
    found = torch.logical_and(df_found_i, df_found_r)
    count_found = torch.count_nonzero(found).item()
    print(count_found)

    # run BB
    # if args.dataset == 'stl10':
    #     attack_bb = LinfinityBrendelBethgeAttack(model, steps=100, clip_min=-1., clip_max=1.)
    # else:
    #     attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    # adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=combined_mask[df_found])
    # adv[df_found] = adv_bb

    # if args.dataset == 'stl10':
    #     import torchvision.transforms as transforms
    #     invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
    #                                                         std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
    #                                    transforms.Normalize(mean=[-0.5, -0.5, -0.5],
    #                                                         std=[1., 1., 1.]),
    #                                    ])
    #     diff_adv = torch.abs(invTrans(adv) - invTrans(cln_data))
    # else:
    #     diff_adv = torch.abs(adv - cln_data)
    # epsilon = torch.amax(diff_adv, dim=(1, 2, 3))
    #
    # pred_adv = predict_from_logits(model(adv))
    # found = torch.logical_and(~torch.eq(true_label, pred_adv), torch.lt(epsilon, args.eps))
    # count_found = torch.count_nonzero(found).item()
    # print("Attack success rate: {:.0%}({}/{})".format(count_found / count_correct, count_found, count_correct))

    true_label = true_label[found]
    cln_data = cln_data[found]
    adv_r = adv_r[found]
    adv_i = adv_i[found]

    pred_cln = pred_cln[found]
    pred_adv_r = predict_from_logits(model(adv_r))
    pred_adv_i = predict_from_logits(model(adv_i))

    if args.dataset in ('stl10', 'imagenet'):
        adv_r = invTrans(adv_r)
        adv_i = invTrans(adv_i)
        cln_data = invTrans(cln_data)

    diff_r = torch.abs(adv_r - cln_data)
    epsilon_r = torch.amax(diff_r, dim=(1, 2, 3))
    diff_i = torch.abs(adv_i - cln_data)
    epsilon_i = torch.amax(diff_i, dim=(1, 2, 3))



    # visualize the results
    # visualize = True
    # if visualize:
    #     idx2name = lambda idx: names[idx]
    #     plt.figure(figsize=(10, 8))
    #     for ii in range(count_found):
    #         # clean image
    #         plt.subplot(3, count_found, ii+1)
    #         _imshow(cln_data[ii])
    #         plt.title("clean \n pred: {}".format(idx2name(pred_cln[ii])))
    #         # adv image
    #         plt.subplot(3, count_found, count_found + ii+1)
    #         _imshow(adv_r[ii])
    #         plt.title("regional adv \n pred: {}".format(idx2name(pred_adv_r[ii])))
    #         # adv difference
    #         plt.subplot(3, count_found, 2 * count_found + ii+1)
    #         _imshow(adv_i[ii])
    #         plt.title("regional imperceptible adv \n pred: {}".format(idx2name(pred_adv_r[ii])))
    #     plt.tight_layout()
    #     plt.show()

    visualize = True
    if visualize:
        idx2name = lambda idx: names[idx]
        plt.figure(figsize=(10, 8))
        # clean image
        plt.subplot(2, 3, 1)
        _imshow(cln_data[0])
        plt.title("clean image \n prediction: {}".format(idx2name(pred_cln[0])))
        plt.subplot(2, 3, 4)
        _imshow(mask_r[0])
        plt.title("regional mask")
        # adv image
        plt.subplot(2, 3, 2)
        _imshow(adv_r[0])
        plt.title("regional \n prediction: {}".format(idx2name(pred_adv_r[0])))
        plt.subplot(2, 3, 5)
        _imshow(diff_r[0] * 5)
        plt.title("perturbation")
        # adv difference
        plt.subplot(2, 3, 3)
        _imshow(adv_i[0])
        plt.title("regional & imperceptible \n prediction: {}".format(idx2name(pred_adv_r[0])))
        plt.subplot(2, 3, 6)
        _imshow(diff_i[0] * 5)
        plt.title("perturbation")
        plt.tight_layout()
        plt.show()