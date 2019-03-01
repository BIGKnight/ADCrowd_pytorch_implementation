import torch
import torch.nn as nn
from deformable_conv2d.deformable_conv2d_wrapper import DeformableConv2DLayer


class BasicDeformableConv2D(nn.Module):
    def __init__(self,
                 stride_h, stride_w,
                 pad_h, pad_w,
                 dilation_h=1, dilation_w=1,
                 num_groups=1,
                 deformable_groups=1,
                 im2col_step=1,
                 no_bias=True,
                 ):
        super(BasicDeformableConv2D, self).__init__()
        self.deformable_conv2d = DeformableConv2DLayer(
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            num_groups,
            deformable_groups,
            im2col_step,
            no_bias)

    def forward(self, x, filter, offset, mask):
        return self.deformable_conv2d.forward(x, filter, offset, mask)


class DeformableInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeformableInceptionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Deformable Layer
        self.deformable_conv_part_1 = BasicDeformableConv2D(1, 1, 1, 1)
        self.deformable_conv_part_2 = BasicDeformableConv2D(1, 1, 2, 2)
        self.deformable_conv_part_3 = BasicDeformableConv2D(1, 1, 3, 3)
        # Deformable parameters
        self.deformable_conv_part_1_filter = None
        self.deformable_conv_part_2_filter = None
        self.deformable_conv_part_3_filter = None
        self.deformable_conv_part_1_offset = None
        self.deformable_conv_part_2_offset = None
        self.deformable_conv_part_3_offset = None
        self.deformable_conv_part_1_mask = None
        self.deformable_conv_part_2_mask = None
        self.deformable_conv_part_3_mask = None

    def forward(self, x):
        part_1 = self.deformable_conv_part_1.forward(
            x,
            self.deformable_conv_part_1_filter,
            self.deformable_conv_part_1_offset,
            self.deformable_conv_part_1_mask
        )
        part_2 = self.deformable_conv_part_2.forward(
            x,
            self.deformable_conv_part_2_filter,
            self.deformable_conv_part_2_offset,
            self.deformable_conv_part_2_mask
        )
        part_3 = self.deformable_conv_part_3.forward(
            x,
            self.deformable_conv_part_3_filter,
            self.deformable_conv_part_3_offset,
            self.deformable_conv_part_3_mask
        )
        output = torch.cat((part_1, part_2, part_3), dim=1)
        return output


class DMENet(nn.Module):
    def __init__(self):
        super(DMENet, self).__init__()

    def forward(self, x):
        return
