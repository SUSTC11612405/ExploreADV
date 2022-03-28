import torch
import numpy as np
from utils import sigma_map


def get_sigma_mask(data):
    sigma = sigma_map(data)
    mask = torch.tensor(np.ceil(sigma))
    return mask