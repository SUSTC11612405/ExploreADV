import torch
import torch.nn as nn
import argparse

import onnx
from onnx2pytorch import ConvertModel

from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
from eval.eval_metric import PerceptualDistance
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
    parser.add_argument('--path_model', type=str, default='./models/mnist_convSmallRELU.onnx', help='path to the trained model')
    parser.add_argument('--eps', type=float, default=1.0, help='max perturbation size on each pixel')
    parser.add_argument('--region', type=str, default='whole', help='whole, top, bottom, left, right, select')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of pixels allowed to perturb')
    parser.add_argument('--imperceptible', action="store_true", help='whether to have imperceptible perturbation')
    parser.add_argument('--n_examples', type=int, default=5, help='number of examples to run')
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
    if args.imperceptible:
        from region_proposal import get_sigma_mask
        masks['sigma'] = get_sigma_mask(cln_data.data)
    if args.ratio != 1.0:
        from region_proposal import get_captum_mask
        masks['importance'] = get_captum_mask(model, cln_data.data, true_label.data)
    combined_mask = get_combined_mask(masks, args.ratio)

    # run attack
    # run Deepfool
    if args.dataset == 'stl10':
        combined_mask *= 2
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
        # adv difference
        plt.subplot(3, count_found, 2 * count_found + ii+1)
        _imshow(diff_adv[ii])
        plt.title("Difference \n epsilon: {:.2}".format(epsilon[ii]))
    plt.tight_layout()
    plt.show()
