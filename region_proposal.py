import torch
import numpy as np
from utils import sigma_map
import shap


def normalize(a):
    axis = (1, 2, 3)
    min_a = np.min(a, axis=axis)[:, None, None, None]
    max_a = np.max(a, axis=axis)[:, None, None, None]
    return (a - min_a) / (max_a - min_a)


def quantile(a, q):
    axis = (1, 2, 3)
    q = np.quantile(a, q, axis=axis)[:, None, None, None]
    return np.where(a > q, a, 0.0)


def get_sigma_mask(data, threshold=0.5):
    sigma = sigma_map(data)
    norm = normalize(sigma)
    mask = quantile(norm, threshold)
    # print(np.max(mask))
    return torch.tensor(mask)


def get_shap_explainer(model, background):
    return shap.DeepExplainer(model, background)


def get_shap_mask(data, explainer, threshold=0.5):
    shap_values, indexes = explainer.shap_values(data, ranked_outputs=1)
    shap = np.abs(shap_values[0])
    norm = normalize(shap)
    mask = quantile(norm, threshold)
    # print(mask.shape)
    # print(np.max(mask))
    return torch.tensor(mask)


def get_combined_mask(data, explainer, threshold=0.5):
    sigma = sigma_map(data)
    shap_values, indexes = explainer.shap_values(data, ranked_outputs=1)
    shap = np.abs(shap_values[0])
    norm = normalize(shap * sigma)
    mask = quantile(norm, threshold)
    # print(mask.shape)
    # print(np.max(mask))
    return torch.tensor(mask)