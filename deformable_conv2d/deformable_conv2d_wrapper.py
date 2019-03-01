import torch
from torch.autograd import Function
from torch.nn import Module
# our module
# 这里有个很坑的地方, 这个完全是pytorch的问题,
import deformable_conv2d_gpu


class DeformableConv2DFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 14:
            print("Wrong parameter number, check your input!")
            return
        input = args[0]
        filter = args[1]
        offset = args[2]
        mask = args[3]
        ctx.stride_h = args[4]
        ctx.stride_w = args[5]
        ctx.pad_h = args[6]
        ctx.pad_w = args[7]
        ctx.dilation_h = args[8]
        ctx.dilation_w = args[9]
        ctx.num_groups = args[10]
        ctx.deformable_groups = args[11]
        ctx.im2col_step = args[12]
        ctx.no_bias = args[13]
        output = deformable_conv2d_gpu.forward(
            input,
            filter,
            offset,
            mask,
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.num_groups,
            ctx.deformable_groups,
            ctx.im2col_step,
            ctx.no_bias)
        print(output)
        ctx.save_for_backward(input, filter, offset, mask)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, filter, offset, mask = ctx.saved_tensors
        return deformable_conv2d_gpu.backward(
            input,
            filter,
            offset,
            mask,
            grad_outputs[0],
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.num_groups,
            ctx.deformable_groups,
            ctx.im2col_step,
            ctx.no_bias)


class DeformableConv2DLayer(Module):
    def __init__(self,
                 stride_h, stride_w,
                 pad_h, pad_w,
                 dilation_h=1, dilation_w=1,
                 num_groups=1,
                 deformable_groups=1,
                 im2col_step=1,
                 no_bias=True,
                 ):
        super(DeformableConv2DLayer, self).__init__()
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

# apply() takes no keyword arguments
    def forward(self,
                inputs,
                filter,
                offset,
                mask):
        return DeformableConv2DFunction.apply(
            inputs,
            filter,
            offset,
            mask,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w,
            self.dilation_h, self.dilation_w,
            self.num_groups,
            self.deformable_groups,
            self.im2col_step,
            self.no_bias)
