import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GroundTruthProcess(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GroundTruthProcess, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.FloatTensor(torch.ones(out_channels, in_channels, kernel_size, kernel_size)).cuda()

    def forward(self, x):
        result = F.conv2d(x, self.kernel, bias=None, stride=self.kernel_size, padding=0)
        return result


def NCHW_to_NHWC_np(images):
    return np.swapaxes(np.swapaxes(images, 1, 2), 2, 3)


def NHWC_to_NCHW_np(images):
    return np.swapaxes(np.swapaxes(images, 2, 3), 1, 2)
