import torch
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def normalize(a):
    axis = (1, 2, 3)
    min_a = np.min(a, axis=axis, keepdims=True)
    max_a = np.max(a, axis=axis, keepdims=True)
    return (a - min_a) / (max_a - min_a)


def quantile(a, q):
    axis = (1, 2, 3)
    q = np.quantile(a, q, axis=axis, keepdims=True)
    return np.where(a > q, 1.0, 0.0)


def topk(a, k):
    N = a.shape[0]
    a_copy = np.reshape(a, (N, -1))
    k_th = np.partition(a_copy, -k, axis=1)[:, -k, None]
    a_copy = np.where(a_copy <= k_th, 0.0, 1.0)
    return np.reshape(a_copy, a.shape)


def sigma_map(x):
    """ creates the sigma-map for the batch x (# samples * channels * width * height)"""
    x = x.numpy()
    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0, :, :, :-1] = x[:, :, 1:]
    t[0, :, :, -1] = x[:, :, -1]
    t[1, :, :, 1:] = x[:, :, :-1]
    t[1, :, :, 0] = x[:, :, 0]
    t[2, :, :, :, :-1] = x[:, :, :, 1:]
    t[2, :, :, :, -1] = x[:, :, :, -1]
    t[3, :, :, :, 1:] = x[:, :, :, :-1]
    t[3, :, :, :, 0] = x[:, :, :, 0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    # sd = np.sqrt(sd)

    return sd


def get_sigma_mask(data):
    np_data = data.detach().cpu().numpy()
    sigma_mask = sigma_map(np_data)
    return sigma_mask


def get_shap_explainer(model, background):
    import shap
    # return shap.GradientExplainer(model, background, local_smoothing=0)
    return shap.DeepExplainer(model, background)


def get_dist2boundary_mask(data, _min=0.0, _max=1.0):
    np_data = data.detach().cpu().numpy()
    return np.where(np_data < 0.5, 0.5 + np_data, 1.5 - np_data)


def get_shap_mask(data, explainer):
    shap_values, indexes = explainer.shap_values(data, ranked_outputs=1)
    shap_mask = np.abs(shap_values[0])
    dist = get_dist2boundary_mask(data)
    shap_mask = shap_mask * dist
    return shap_mask


def get_gradcamplusplus_mask(model, data, target_layers):
    from pytorch_grad_cam import GradCAMPlusPlus
    targets = None
    with GradCAMPlusPlus(model=model,
                target_layers=target_layers,
                use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=data,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
        return grayscale_cam


def get_gradcam_mask(model, data, target_layers):
    from pytorch_grad_cam import GradCAM
    targets = None
    with GradCAM(model=model,
                target_layers=target_layers,
                use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=data,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
        return grayscale_cam


def get_captum_mask(model, data, label, correction=True):
    from captum.attr import IntegratedGradients, NoiseTunnel

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=label, **kwargs)
        return tensor_attributions

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig = attribute_image_features(nt, data, baselines=data * 0, nt_type='smoothgrad_sq',
                                       nt_samples=10, stdevs=0.2)
    # attr_ig, delta = attribute_image_features(ig, data, baselines=data * 0, return_convergence_delta=True)
    captum_mask = attr_ig.detach().cpu().numpy()
    if correction:
        dist = get_dist2boundary_mask(data)
        captum_mask *= dist
    return captum_mask


def get_nbyn_mask(data, n, x, y):
    np_data = data.detach().cpu().numpy()
    mask = np.zeros_like(np_data)
    mask[:, :, x:x+n, y:y+n] = 1.0
    return torch.tensor(mask, dtype=torch.float32).to(device)


def get_region_mask(data, region):
    np_data = data.detach().cpu().numpy()
    rows, cols = data.shape[2:]
    region_mask = np.ones_like(np_data)
    if region == 'top':
        region_mask[:, :, rows // 2:, :] = 0.0
    elif region == 'bottom':
        region_mask[:, :, :rows // 2, :] = 0.0
    elif region == 'left':
        region_mask[:, :, :, cols // 2:] = 0.0
    elif region == 'right':
        region_mask[:, :, :, :cols // 2] = 0.0
    return region_mask


def get_combined_mask(masks, ratio):
    mask = masks['region']
    if 'sigma' in masks:
        mask *= masks['sigma']
    if 'importance' in masks:
        masks['importance'] *= masks['region']
    if ratio < 1.0:
        mask *= quantile(masks['importance'], 1.0 - ratio)
    elif ratio > 1.0:
        mask *= topk(masks['importance'], int(ratio))
    return torch.tensor(mask, dtype=torch.float32).to(device)
