import numpy as np
import torch
import torch.nn as nn

from utils import predict_from_logits
from dataloader import get_mnist_test_loader, get_cifar10_test_loader
from utils import _imshow
from region_proposal import get_sigma_mask
import onnx
from onnx2pytorch import ConvertModel
from attacks import LinfPGDAttack, DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
import matplotlib.pyplot as plt

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# path_to_onnx_model = "./models/convSmallRELU__Point.onnx"

# path_to_onnx_model = "./models/mnist_relu_6_100.onnx"
path_to_onnx_model = "./models/mnist_relu_9_200.onnx"
# path_to_onnx_model = "./models/cifar10_2_255.onnx"

onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model, experimental=True)

model = pytorch_model
model.to(device)
model.eval()

print(model)

batch_size = 5
loader = get_mnist_test_loader(batch_size=batch_size)
# loader = get_cifar10_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)
# print(cln_data)
print(true_label)

mask = get_sigma_mask(cln_data.data)

# adversary = LinfPGDAttack(
#     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.63,
#     nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
#     targeted=False)

adversary = DeepfoolLinfAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.58)

adv_untargeted = adversary.perturb(cln_data, true_label, mask)

# apply the Brendel & Bethge attack
attack = LinfinityBrendelBethgeAttack(model, steps=100)

BB_adversarials = attack.perturb(cln_data, adv_untargeted, mask=mask)


# diff_pgd_adv = np.abs(adv_untargeted.numpy() - cln_data.numpy())
# print('PGD_adv:\t', np.max(diff_pgd_adv, axis=(1,2,3)))

diff_df_adv = np.abs(adv_untargeted.numpy() - cln_data.numpy())
epsilon_df = np.max(diff_df_adv, axis=(1,2,3))
print('Deepfool_adv:\t', epsilon_df)

diff_bb = np.abs(BB_adversarials.numpy() - cln_data.numpy())
epsilon_bb = np.max(diff_bb, axis=(1,2,3))
print('BB refined DF:\t', epsilon_bb)

### Visualization of attacks

pred_cln = predict_from_logits(model(cln_data))
# pred_pgd = predict_from_logits(model(adv_untargeted))
pred_df = predict_from_logits(model(adv_untargeted))
pred_bb = predict_from_logits(model(BB_adversarials))

plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(4, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size)
    _imshow(mask[ii])
    plt.title("mask of {}:".format(pred_cln[ii]))
    plt.subplot(4, batch_size, ii + 1 + 2 * batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("Deepfool \n pred: {} \n epsilon: {:.2}".format(
        pred_df[ii], epsilon_df[ii]))
    plt.subplot(4, batch_size, ii + 1 + 3 * batch_size)
    _imshow(BB_adversarials[ii])
    plt.title("BB refined \n pred: {} \n epsilon: {:.2}".format(
        pred_bb[ii], epsilon_bb[ii]))

plt.tight_layout()
plt.show()