import torch
from .differential_color_function import rgb2lab_diff, ciede2000_diff
from .ssim import SSIM


class PerceptualDistance(object):
    def __init__(self, dataset):

        self.dataset = dataset

        self.SSIM = SSIM()

        self.avg = {'l0': 0, 'l2': 0, 'l_inf': 0, 'ssim': 0}
        self.sum = {'l0': 0, 'l2': 0, 'l_inf': 0, 'ssim': 0}

        if self.dataset != 'mnist':
            self.avg['CIEDE2000'] = 0
            self.sum['CIEDE2000'] = 0

        self.count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cal_perceptual_distances(self, references, perturbed):
        distance = {}
        # l_p norm
        N = references.size(0)
        noise = (perturbed - references).flatten(start_dim=1)
        distance['l0'] = torch.sum(torch.count_nonzero(noise, dim=-1)) / N
        distance['l2'] = torch.sum(torch.pow(torch.norm(noise, p=2, dim=-1), 2)) / N
        distance['l_inf'] = torch.sum(torch.norm(noise, p=float('inf'), dim=-1)) / N

        # SSIM
        distance['ssim'] = self.cal_ssim(references, perturbed) / N

        if self.dataset != 'mnist':
            # perceptual color distance
            distance['CIEDE2000'] = self.cal_color_distance(references, perturbed) / N

        return distance

    def cal_color_distance(self, references, perturbed):
        reference_lab = rgb2lab_diff(references, self.device)
        perturbed_lab = rgb2lab_diff(perturbed, self.device)
        color_distance_map = ciede2000_diff(reference_lab, perturbed_lab, self.device)
        color_distance_map = color_distance_map.flatten(start_dim=1)
        norm = torch.norm(color_distance_map, dim=-1)
        return torch.sum(norm)

    def cal_ssim(self, references, perturbed):
        ret = self.SSIM
        ssim = torch.zeros(references.shape[0])
        for i in range(references.shape[0]):
            ssim[i] = ret.forward(references[i].unsqueeze(0) * 255, perturbed[i].unsqueeze(0) * 255)
        return torch.sum(ssim)

    def update(self, distance, n=1):

        for k, v in distance.items():
            self.sum[k] += v * n

        self.count += n

        for k, v in self.sum.items():
            self.avg[k] += v / self.count

    def print_metric(self):
        info_str = ", ".join(["{}: {:.2f}".format(k, v) for k, v in self.avg.items()])
        print(info_str)
