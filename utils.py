import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class GroundTruthProcess(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GroundTruthProcess, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.FloatTensor(torch.ones(out_channels, in_channels, kernel_size, kernel_size)).cuda()

    def forward(self, x):
        result = F.conv2d(x, self.kernel, bias=None, stride=self.kernel_size, padding=0)
        return result


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()
