import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel
from dataloader import get_mnist_test_loader, get_stl10_test_loader, get_cifar10_test_loader
from utils import predict_from_logits


def calculate_accuracy(dataset, path):
    if dataset == 'mnist':
        onnx_model = onnx.load(path)
        model = ConvertModel(onnx_model, experimental=True)
        loader, size = get_mnist_test_loader(batch_size=n_examples, get_size=True)
    elif dataset == 'cifar10':
        onnx_model = onnx.load(path)
        model = ConvertModel(onnx_model, experimental=True)
        loader, size = get_cifar10_test_loader(batch_size=n_examples, get_size=True)
    elif dataset == 'stl10':
        from stl10 import stl10
        model = stl10(32, pretrained=True)
        loader, size = get_stl10_test_loader(batch_size=n_examples, get_size=True)
    else:
        error = "Only mnist, cifar10, stl10 data supported"
        raise NotImplementedError(error)
    count = 0
    for cln_data, true_label in loader:
        count += torch.sum(torch.eq(true_label, predict_from_logits(model(cln_data)))).item()
    print(count / size)


if __name__ == '__main__':
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset = 'mnist'
    # path_model = './models/mnist_relu_9_200.onnx'  # 0.9495
    # path_model = './models/convSmallRELU__Point.onnx'  # 0.9825
    dataset = 'cifar10'
    path_model = "./models/cifar10_relu_6_500.onnx"  # 0.5598
    # path_model = "./models/convMedGSIGMOID__Point.onnx"  # 0.5495
    path_model = "./models/convBigRELU__DiffAI_cifar10.onnx"  # 0.5144
    # path_model = "./models/ResNet18_PGD_cifar10.onnx"  # 0.8152
    # dataset = 'stl10'  # 0.772
    n_examples = 100
    calculate_accuracy(dataset, path_model)



