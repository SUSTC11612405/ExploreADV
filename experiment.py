#%%

import torch
import torch.nn as nn
import numpy as np

from utils import predict_from_logits
import torchvision.transforms as transforms
from dataloader import get_mnist_test_loader, get_cifar10_test_loader, get_stl10_test_loader, \
    get_cifar10_normalized_test_loader
from attacks import DeepfoolLinfAttack, LinfinityBrendelBethgeAttack

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


import onnx
from onnx2pytorch import ConvertModel

# path_to_onnx_model = "./models/convSmallRELU__Point.onnx"
# path_to_onnx_model = "./models/mnist_relu_9_200.onnx"

# path_to_onnx_model = "./models/cifar10_relu_6_500.onnx"
path_to_onnx_model = "./models/convMedGSIGMOID__Point.onnx"
# path_to_onnx_model = "./models/convBigRELU__DiffAI_cifar10.onnx"
# path_to_onnx_model = "./models/ResNet18_PGD_cifar10.onnx"

onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model, experimental=True)
# from stl10 import stl10
# pytorch_model = stl10(32, pretrained=True)

model = pytorch_model
model.to(device)
model.eval()

print(model)


def cauculate_linf_distance(adv, cln):
    # invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
    #                                                     std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
    #                                transforms.Normalize(mean=[-0.5, -0.5, -0.5],
    #                                                     std=[1., 1., 1.]),
    #                                ])
    # diff_df_adv = np.abs(invTrans(adv).numpy() - invTrans(cln).numpy())
    diff_df_adv = np.abs(adv.numpy() - cln.numpy())
    epsilon_df = np.max(diff_df_adv, axis=(1, 2, 3))
    return epsilon_df


count = {'df+bb': 0}
sum_dist = {'df+bb': 0}


def get_adv_count_and_avg_dist(name, data, label):

    if name == 'df+bb':
        attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
        starting_points = attack_df.perturb(data, label)
        attack = LinfinityBrendelBethgeAttack(model, steps=100)
        adv = attack.perturb(data, starting_points)

    found = (predict_from_logits(model(adv)) != label).numpy()
    dist = cauculate_linf_distance(adv, data)

    return np.sum(found), np.sum(dist[found])

total = 0
batch_size = 10
# loader = get_mnist_test_loader(batch_size=batch_size)
loader = get_cifar10_test_loader(batch_size=batch_size)
# loader = get_stl10_test_loader(batch_size=batch_size)

for cln_data, true_label in loader:
    correct = predict_from_logits(model(cln_data)) == true_label
    cln_data = cln_data[correct]
    true_label = true_label[correct]
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    if cln_data.shape[0] == 0:
        continue

    for k in count:
        _count, _sum = get_adv_count_and_avg_dist(k, cln_data.data, true_label)
        count[k] += _count
        sum_dist[k] += _sum

    total += len(cln_data)
    print(total)
    if total > 100:
        break

for k in count:
    sum_dist[k] /= count[k]

print(count)
print(sum_dist)
