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
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DeformableInceptionModule, self).__init__()
        # Deformable Layer
        self.deformable_conv_part_1 = BasicDeformableConv2D(1, 1, 1, 1)
        self.deformable_conv_part_2 = BasicDeformableConv2D(1, 1, 2, 2)
        self.deformable_conv_part_3 = BasicDeformableConv2D(1, 1, 3, 3)
        # Deformable parameters generators
        self.offset_generator_1 = nn.Conv2d(in_channels, 3 * 3 * 2, kernel_size=3, padding=1)
        self.offset_generator_2 = nn.Conv2d(in_channels, 5 * 5 * 2, kernel_size=5, padding=2)
        self.offset_generator_3 = nn.Conv2d(in_channels, 7 * 7 * 2, kernel_size=7, padding=3)
        self.mask_generator_1 = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, padding=1)
        self.mask_generator_2 = nn.Conv2d(in_channels, 5 * 5, kernel_size=5, padding=2)
        self.mask_generator_3 = nn.Conv2d(in_channels, 7 * 7, kernel_size=7, padding=3)
        self.deformable_conv_part_2_filter = None
        self.deformable_conv_part_3_filter = None
        # filter initialization(Xaviar)
        self.filter_1 = nn.init.xavier_uniform_(
            torch.zeros(out_channels, in_channels, 3, 3, dtype=torch.float32),
            gain=1
        )
        self.filter_2 = nn.init.xavier_uniform_(
            torch.zeros(out_channels, in_channels, 5, 5, dtype=torch.float32),
            gain=1
        )
        self.filter_3 = nn.init.xavier_uniform_(
            torch.zeros(out_channels, in_channels, 7, 7, dtype=torch.float32),
            gain=1
        )

    def forward(self, x):
        # generate the offset and mask
        offset_1 = self.offset_generator_1(x)
        offset_2 = self.offset_generator_2(x)
        offset_3 = self.offset_generator_3(x)
        mask_1 = self.mask_generator_1(x)
        mask_2 = self.mask_generator_2(x)
        mask_3 = self.mask_generator_3(x)
        # do the deformable convolution
        part_1 = self.deformable_conv_part_1.forward(
            x,
            self.filter_1,
            offset_1,
            mask_1
        )
        part_2 = self.deformable_conv_part_2.forward(
            x,
            self.filter_2,
            offset_2,
            mask_2
        )
        part_3 = self.deformable_conv_part_3.forward(
            x,
            self.filter_3,
            offset_3,
            mask_3
        )
        # concat
        output = torch.cat((part_1, part_2, part_3), dim=1)
        return output


class DMENet(nn.Module):
    def __init__(self):
        super(DMENet, self).__init__()
        self.front_end = None
        self.back_end = nn.Sequential(
            DeformableInceptionModule(512, 256),
            nn.Conv2d(256 * 3, 256, kernel_size=1),
            DeformableInceptionModule(256, 128),
            nn.Conv2d(128 * 3, 128, kernel_size=1),
            DeformableInceptionModule(128, 64),
            nn.Conv2d(64 * 3, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.front_end(x)
        out = self.back_end(features)
        return out
