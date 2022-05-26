import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import onnx
from onnx2pytorch import ConvertModel

from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
from eval.eval_metric import PerceptualDistance
from region_proposal import get_region_mask, get_combined_mask
from utils import predict_from_logits, _imshow, _imshow_diff
import time

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


def get_epsilon(cln_data, true_label, topk, n=10):
    from region_proposal import get_nbyn_mask, get_captum_mask
    mask = get_captum_mask(model, cln_data, true_label)

    w, h = cln_data.data.shape[-2:]

    maxk = max(topk)
    pos_score = []

    start_time = time.time()

    for i in range(w - n):
        for j in range(h - n):
            mask_sum = np.sum(mask[:, :, i:i + n, j:j + n])
            pos_score.append(((i, j), mask_sum))

    pos_score.sort(key=lambda x: x[1], reverse=True)
    pos = [ps[0] for ps in pos_score[:maxk]]

    epsilons = {}
    times = {}
    smallest = 1.0
    for i in range(maxk):
        mask_n = get_nbyn_mask(cln_data.data, n, pos[i][0], pos[i][1])
        if dataset == 'stl10':
            mask_n *= 2
            attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0,
                                           clip_min=-1., clip_max=1.)
        else:
            attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
        adv = attack_df.perturb(cln_data, true_label, mask=mask_n)

        pred_df = predict_from_logits(model(adv))
        df_found = ~torch.eq(true_label, pred_df)
        count_df_found = torch.count_nonzero(df_found).item()
        if count_df_found == 0:
            if i + 1 in topk:
                epsilons[i + 1] = smallest
                times[i + 1] = time.time() - start_time
            continue

        # run BB
        if dataset == 'stl10':
            attack_bb = LinfinityBrendelBethgeAttack(model, steps=100, clip_min=-1., clip_max=1.)
        else:
            attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
        adv_bb = attack_bb.perturb(cln_data, adv, mask=mask_n)
        if dataset == 'stl10':
            invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                           transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                std=[1., 1., 1.]),
                                           ])
            diff = torch.abs(invTrans(adv_bb) - invTrans(cln_data))
        else:
            diff = torch.abs(adv_bb - cln_data)
        epsilon = torch.amax(diff, dim=(1, 2, 3))[0].item()
        if epsilon < smallest:
            smallest = epsilon
        if i+1 in topk:
            epsilons[i+1] = smallest
            times[i+1] = time.time()-start_time
    return epsilons, times

if __name__ == '__main__':

    # dataset = 'mnist'
    # path_model = './models/mnist_relu_9_200.onnx'
    # path_model = './models/mnist_convSmallRELU.onnx'
    dataset = 'cifar10'
    # path_model = "./models/cifar10_relu_6_500.onnx"
    # path_model = "./models/cifar10_convMedGSIGMOID.onnx"
    # path_model = "./models/cifar10_convBigRELU_DiffAI.onnx"
    path_model = "../models/cifar10_ResNet18_PGD.onnx"
    # dataset = 'stl10'
    n_examples = 1
    eps = 1.0
    write_file_path = '../results/mnist_convSmallRELU_topk.csv'

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(dataset, path_model)

    topk = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]

    # with open(write_file_path, mode='a') as file:
    #     file.write("index," +
    #                ",".join(["top{}".format(k) for k in topk]) + "\n")

    # load data
    loader, names = get_dataloader_with_names(dataset, n_examples)
    index = 0
    count = 0

    sum_times = {i: 0 for i in topk}

    for cln_data, true_label in loader:
        # if index <= 5:
        #     index += 1
        #     continue
        if count == 10:
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

        epsilons, times = get_epsilon(cln_data, true_label, topk)

        for k in topk:
            sum_times[k] += times[k]
        # print(epsilons)
        # with open(write_file_path, mode='a') as file:
        #     file.write(str(index) + "," +
        #                ",".join(["{:.4}".format(epsilons[i]) for i in topk])+"\n")
        index += 1
        count += 1

    for k in topk:
        sum_times[k] /= 10
    with open('../results/topk_timecost.csv', mode='a') as file:
        file.write("C4," +
                   ",".join(["{:.4}".format(sum_times[i]) for i in topk])+"\n")
    print(sum_times)
