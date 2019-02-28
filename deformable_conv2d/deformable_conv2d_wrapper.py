import torch
from torch.autograd import Function
from torch.nn import Module
# our module
# 这里有个很坑的地方, 这个完全是pytorch的问题,
import deformable_conv2d_gpu


class DeformableConv2DFunction(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        return

    @staticmethod
    def backward(ctx, *grad_outputs):
        return


    # int stride_h,
    # int stride_w,
    # int num_groups,
    # int deformable_groups,
    # int im2col_step,
    # bool no_bias,
    # int pad_h,
    # int pad_w,
    # int dilation_h,
    # int dilation_w

class DeformableConv2DLayer(Module):
    def __init__(self,
                 input,
                 filter,
                 offset,
                 mask,
                 stride_h, stride_w,
                 pad_h, pad_w,
                 dilation_h=1, dilation_w=1,
                 num_groups=1,
                 deformable_groups=1,
                 im2col_step=1,
                 no_bias=True,
                 ):
        super(DeformableConv2DLayer, self).__init__()
        self.input = input
        self.filter = filter
        self.offset = offset
        self.mask = mask
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.no_bias = no_bias

    def forward(self, input, state):
        return
