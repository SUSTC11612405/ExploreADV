import numpy as np
import torch
import torch.nn as nn

from utils import predict_from_logits
from dataloader import get_mnist_test_loader, get_mnist_train_loader
from dataloader import get_cifar10_test_loader, get_cifar10_train_loader
from dataloader import get_stl10_train_loader, get_stl10_test_loader
from utils import _imshow
from region_proposal import get_sigma_mask, get_shap_mask, get_shap_explainer, get_combined_mask
import onnx
from onnx2pytorch import ConvertModel
from attacks import LinfPGDAttack, DeepfoolLinfAttack, LinfinityBrendelBethgeAttack
import matplotlib.pyplot as plt

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# path_to_onnx_model = "./models/convSmallRELU__Point.onnx"
# path_to_onnx_model = "./models/mnist_relu_6_100.onnx"
# path_to_onnx_model = "./models/mnist_relu_9_200.onnx"

# path_to_onnx_model = "./models/cifar10_2_255.onnx"
# path_to_onnx_model = "./models/convBigRELU__DiffAI_cifar10.onnx"
# path_to_onnx_model = "./models/ResNet18_PGD_cifar10.onnx"

# onnx_model = onnx.load(path_to_onnx_model)

from stl10 import stl10
pytorch_model = stl10(32, pretrained=True)
# pytorch_model = ConvertModel(onnx_model, experimental=True)

model = pytorch_model
model.to(device)
model.eval()

print(model)

# shap_loader = get_mnist_train_loader(batch_size=100)
# shap_loader = get_cifar10_train_loader(batch_size=100)
shap_loader = get_stl10_train_loader(batch_size=100)
for background, _ in shap_loader:
    break
background = background.to(device)

batch_size = 3
# loader = get_mnist_test_loader(batch_size=batch_size)
# loader = get_cifar10_test_loader(batch_size=batch_size)
loader = get_stl10_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    if (predict_from_logits(model(cln_data)) == true_label).all():
        break
cln_data, true_label = cln_data.to(device), true_label.to(device)
# print(cln_data)
# print(true_label)

e = get_shap_explainer(model, background)
mask_I = get_shap_mask(cln_data.data, e, 0.5)
mask_V = get_sigma_mask(cln_data.data, 0.5)
mask_C = get_combined_mask(cln_data.data, e, 0.5)


def get_adv(data, label, mask):
    attack_df = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.0)
    starting_points = attack_df.perturb(cln_data, label, mask=mask)
    attack = LinfinityBrendelBethgeAttack(model, steps=100)
    adv = attack.perturb(cln_data, starting_points, mask=mask)

    diff_adv = np.abs(adv.numpy() - data.numpy())
    epsilon = np.max(diff_adv, axis=(1, 2, 3))

    return adv, epsilon


adv, epsilon = get_adv(cln_data, true_label, None)
adv_I, epsilon_I = get_adv(cln_data, true_label, mask_I)
adv_V, epsilon_V = get_adv(cln_data, true_label, mask_V)
adv_C, epsilon_C = get_adv(cln_data, true_label, mask_C)




# MNIST
# names = list(range(10))
# Cifar10
names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

idx2name = lambda indexes: [names[i] for i in indexes]

pred_cln = idx2name(predict_from_logits(model(cln_data)))
pred_adv = idx2name(predict_from_logits(model(adv)))
pred_I = idx2name(predict_from_logits(model(adv_I)))
pred_V = idx2name(predict_from_logits(model(adv_V)))
pred_C = idx2name(predict_from_logits(model(adv_C)))

plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    # clean image and all-pixel adversarial
    plt.subplot(4, 2 * batch_size, 2 * ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(4, 2 * batch_size, 2 * ii + 2)
    _imshow(adv[ii])
    plt.title("adversarial \n pred: {} \n epsilon: {:.2}".format(pred_adv[ii], epsilon[ii]))

    # importance map and 50%-pixel adversarial
    plt.subplot(4, 2 * batch_size, 2 * batch_size + 2 * ii + 1)
    _imshow(mask_I[ii])
    plt.title("importance map")
    plt.subplot(4, 2 * batch_size, 2 * batch_size + 2 * ii + 2)
    _imshow(adv_I[ii])
    plt.title("adversarial \n pred: {} \n epsilon: {:.2}".format(pred_I[ii], epsilon_I[ii]))

    # importance map and 50%-pixel adversarial
    plt.subplot(4, 2 * batch_size, 4 * batch_size + 2 * ii + 1)
    _imshow(mask_V[ii])
    plt.title("variance map")
    plt.subplot(4, 2 * batch_size, 4 * batch_size + 2 * ii + 2)
    _imshow(adv_V[ii])
    plt.title("adversarial \n pred: {} \n epsilon: {:.2}".format(pred_V[ii], epsilon_V[ii]))

    # importance map and 50%-pixel adversarial
    plt.subplot(4, 2 * batch_size, 6 * batch_size + 2 * ii + 1)
    _imshow(mask_C[ii])
    plt.title("combined map")
    plt.subplot(4, 2 * batch_size, 6 * batch_size + 2 * ii + 2)
    _imshow(adv_C[ii])
    plt.title("adversarial \n pred: {} \n epsilon: {:.2}".format(pred_C[ii], epsilon_C[ii]))

plt.tight_layout()
plt.show()