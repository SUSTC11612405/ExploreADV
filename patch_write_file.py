import torch
import torch.nn as nn
import numpy as np

import onnx
from onnx2pytorch import ConvertModel

from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
from region_proposal import get_nbyn_mask, get_captum_mask
from utils import predict_from_logits
import time
from tqdm import tqdm


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
    path_model = "./models/convMedGSIGMOID__Point.onnx"
    # path_model = "./models/convBigRELU__DiffAI_cifar10.onnx"
    # path_model = "./models/ResNet18_PGD_cifar10.onnx"
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
    index = 0
    count = 42
    for cln_data, true_label in loader:
        if index <= 66:
            index += 1
            continue
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

        importance_mask = get_captum_mask(model, cln_data, true_label)

        w, h = cln_data.data.shape[-2:]
        n = 10
        results = []
        closest = 1
        pos_closest = (0, 0)
        largest = 0
        pos_largest = (0, 0)

        start_time = time.time()

        for i in tqdm(range(w - n)):
            for j in range(h - n):
                mask = get_nbyn_mask(cln_data.data, n, i, j)
                # run attack
                # run Deepfool
                attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
                adv = attack_df.perturb(cln_data, true_label, mask=mask)

                pred_df = predict_from_logits(model(adv))
                df_found = ~torch.eq(true_label, pred_df)
                count_df_found = torch.count_nonzero(df_found).item()
                if count_df_found == 0:
                    continue

                # run BB
                attack_bb = LinfinityBrendelBethgeAttack(model, steps=100)
                adv_bb = attack_bb.perturb(cln_data[df_found], adv[df_found], mask=mask[df_found])
                adv[df_found] = adv_bb

                diff_adv = torch.abs(adv - cln_data)
                epsilon = torch.amax(diff_adv, dim=(1, 2, 3))

                pred_adv = predict_from_logits(model(adv))
                found = torch.logical_and(~torch.eq(true_label, pred_adv), torch.lt(epsilon, eps))
                count_found = torch.count_nonzero(found).item()
                if count_found == 0:
                    continue

                true_label = true_label[found]
                cln_data = cln_data[found]
                adv = adv[found]
                diff_adv = diff_adv[found]
                epsilon = epsilon[found]
                pred_cln = pred_cln[found]
                pred_adv = pred_adv[found]

                importance_sum = np.sum(importance_mask[:, :, i:i + n, j:j + n])
                if importance_sum > largest:
                    pos_largest = (i, j)
                    largest = importance_sum
                    adv_shap = adv
                    diff_shap = diff_adv

                epsilon = epsilon[0].item()

                if epsilon < closest:
                    pos_closest = (i, j)
                    closest = epsilon
                    adv_smallest = adv
                    diff_smallest = diff_adv

                results.append((epsilon, importance_sum))

        print("Time cost: ", time.time()-start_time)
        results = sorted(results)

        with open('cifar10_convMedGSIGMOID.csv', mode='a') as file:
            file.write("{},{:.4},{:.4},{:.4},{:.4},{:.4}\n"
                       .format(index,
                               sum(1 for r in results) / ((w-n)*(h-n)),
                               sorted(results, key=lambda x: x[1], reverse=True)[0][0],
                               results[0][0],
                               sum(r[0] for r in results)/sum(1 for r in results),
                               results[-1][0]))
        index += 1
        count += 1
