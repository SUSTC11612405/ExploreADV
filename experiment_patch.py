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

    # dataset = 'mnist'
    # path_model = './models/mnist_relu_9_200.onnx'
    # path_model = './models/convSmallRELU__Point.onnx'
    dataset = 'cifar10'
    # path_model = "./models/cifar10_relu_6_500.onnx"
    # path_model = "./models/convMedGSIGMOID__Point.onnx"
    # path_model = "./models/convBigRELU__DiffAI_cifar10.onnx"
    path_model = "./models/ResNet18_PGD_cifar10.onnx"
    # dataset = 'stl10'
    n_examples = 1
    eps = 1.0

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(dataset, path_model)

    # load data
    loader, names = get_dataloader_with_names(dataset, n_examples)
    count = 0
    for cln_data, true_label in loader:
        if count == 1:
            break
        count += 1
    cln_data, true_label = cln_data.to(device), true_label.to(device)

    # calculate raw precision
    pred_cln = predict_from_logits(model(cln_data))
    correct = torch.eq(true_label, pred_cln)
    count_correct = torch.count_nonzero(correct).item()
    print("Raw model precision: {:.0%}({}/{})".format(count_correct / n_examples, count_correct, n_examples))
    cln_data = cln_data[correct]
    true_label = true_label[correct]
    pred_cln = pred_cln[correct]

    # generate masks
    # from draw import region_selector
    # mask = torch.tensor(region_selector(cln_data))

    from region_proposal import get_nbyn_mask, get_captum_mask
    importance_mask = get_captum_mask(model, cln_data, true_label)

    w, h = cln_data.data.shape[-2:]
    n = 10
    results = []
    closest = 1
    pos_closest = (0, 0)
    largest = 0
    pos_largest = (0, 0)

    import time
    start_time = time.time()

    for i in range(w - n):
        for j in range(h - n):
            shap_sum = np.sum(importance_mask[:, :, i:i + n, j:j + n])
            if shap_sum > largest:
                pos_largest = (i, j)
                largest = shap_sum
    print(pos_largest)
    mask = get_nbyn_mask(cln_data.data, n, pos_largest[0], pos_largest[1])

    attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    adv = attack_df.perturb(cln_data, true_label, mask=mask)

    pred_df = predict_from_logits(model(adv))
    df_found = ~torch.eq(true_label, pred_df)
    count_df_found = torch.count_nonzero(df_found).item()

    # run BB
    attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=mask[df_found])

    diff_df = torch.abs(adv - cln_data)
    diff_bb = torch.abs(adv_bb - cln_data)
    epsilon_df = torch.amax(diff_df, dim=(1, 2, 3))[0].item()
    epsilon_bb = torch.amax(diff_bb, dim=(1, 2, 3))[0].item()
    print(epsilon_df, epsilon_bb)

    # print(predict_from_logits(model(adv)), predict_from_logits(model(adv_bb)))
    #
    # idx2name = lambda idx: names[idx]
    # plt.figure(figsize=(10, 8))
    # # clean image
    # plt.subplot(2, 2, 1)
    # _imshow(cln_data[0])
    # plt.title("clean")
    # # mask
    # plt.subplot(2, 2, 2)
    # _imshow(mask[0])
    # plt.title("mask")
    # # deepfool
    # plt.subplot(2, 2, 3)
    # _imshow(adv[0])
    # plt.title("adv \n epsilon: {:.2}".format(epsilon_df))
    # # bb
    # plt.subplot(2, 2, 4)
    # _imshow(adv_bb[0])
    # plt.title("optimized adv \n epsilon: {:.2}".format(epsilon_bb))
    # plt.tight_layout()
    # plt.show()

    # for i in range(w - n):
    #     for j in range(h - n):
    #         mask = get_nbyn_mask(cln_data.data, n, i, j)
    #         # run attack
    #         # run Deepfool
    #         attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    #         adv = attack_df.perturb(cln_data, true_label, mask=mask)
    #
    #         pred_df = predict_from_logits(model(adv))
    #         df_found = ~torch.eq(true_label, pred_df)
    #         count_df_found = torch.count_nonzero(df_found).item()
    #         # print("Deepfool attack success rate: {:.0%}({}/{})".format(count_df_found / count_correct, count_df_found, count_correct))
    #         if count_df_found == 0:
    #             print("Deepfool failed to find any")
    #             continue
    #
    #         # run BB
    #         attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
    #         adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=mask[df_found])
    #         adv[df_found] = adv_bb
    #
    #         diff_adv = torch.abs(adv - cln_data)
    #         epsilon = torch.amax(diff_adv, dim=(1, 2, 3))
    #
    #         pred_adv = predict_from_logits(model(adv))
    #         found = torch.logical_and(~torch.eq(true_label, pred_adv), torch.lt(epsilon, eps))
    #         count_found = torch.count_nonzero(found).item()
    #         # print("Attack success rate: {:.0%}({}/{})".format(count_found / count_correct, count_found, count_correct))
    #         if count_found == 0:
    #             print("BB failed to find any")
    #             continue
    #
    #         true_label = true_label[found]
    #         cln_data = cln_data[found]
    #         adv = adv[found]
    #         diff_adv = diff_adv[found]
    #         epsilon = epsilon[found]
    #         pred_cln = pred_cln[found]
    #         pred_adv = pred_adv[found]
    #
    #         # calculate metrics
    #         # PerD = PerceptualDistance(dataset)
    #         # distance = PerD.cal_perceptual_distances(cln_data, adv)
    #         # PerD.update(distance, adv.size(0))
    #         # PerD.print_metric()
    #
    #         importance_sum = np.sum(importance_mask[:, :, i:i + n, j:j + n])
    #         if importance_sum > largest:
    #             pos_largest = (i, j)
    #             largest = importance_sum
    #             adv_shap = adv
    #             diff_shap = diff_adv
    #
    #         epsilon = epsilon[0].item()
    #
    #         if epsilon < closest:
    #             pos_closest = (i, j)
    #             closest = epsilon
    #             adv_smallest = adv
    #             diff_smallest = diff_adv
    #
    #         print(epsilon, importance_sum)
    #         results.append((epsilon, importance_sum))
    #
    print("Time cost: ", time.time()-start_time)
    # results = sorted(results)
    #
    # print(" {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |"
    #       .format(sum(1 for r in results) / ((w-n)*(h-n)),
    #               sorted(results, key=lambda x: x[1], reverse=True)[0][0],
    #               results[0][0],
    #               sum(r[0] for r in results)/sum(1 for r in results),
    #               results[-1][0]))
    #
    # print("shap: ", sorted(results, key=lambda x: x[1], reverse=True)[0][0])
    #
    # print(pos_closest)
    # print(pos_largest)

    # idx2name = lambda idx: names[idx]
    # plt.figure(figsize=(10, 8))
    # # clean image
    # plt.subplot(3, 2, 1)
    # _imshow(cln_data[0])
    # plt.title("clean")
    # # shap image
    # plt.subplot(3, 2, 3)
    # _imshow(adv_shap[0])
    # plt.title("adversarial \n shap")
    # # shap difference
    # plt.subplot(3, 2, 4)
    # _imshow(diff_shap[0])
    # plt.title("Difference \n shap")
    # # smallest image
    # plt.subplot(3, 2, 5)
    # _imshow(adv_smallest[0])
    # plt.title("adversarial \n smallest")
    # # smallest difference
    # plt.subplot(3, 2, 6)
    # _imshow(diff_smallest[0])
    # plt.title("Difference \n smallest")
    # plt.tight_layout()
    # plt.show()
