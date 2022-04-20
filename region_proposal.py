import torch
import numpy as np
import shap


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
    sigma_mask = sigma_map(data)
    return sigma_mask


def get_shap_explainer(model, background):
    return shap.DeepExplainer(model, background)


def get_shap_mask(data, explainer):
    shap_values, indexes = explainer.shap_values(data, ranked_outputs=1)
    shap_mask = np.abs(shap_values[0])
    return shap_mask


def get_region_mask(data, region):
    rows, cols = data.shape[2:]
    region_mask = np.ones_like(data)
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
    if ratio < 1.0:
        mask *= quantile(masks['shap'], 1.0 - ratio)
        # random_mask = np.random.rand(*masks['region'].shape)
        # mask *= quantile(random_mask, 1.0 - ratio)
    elif ratio > 1.0:
        mask *= topk(masks['shap'], int(ratio))
    return torch.tensor(mask, dtype=torch.float32)
