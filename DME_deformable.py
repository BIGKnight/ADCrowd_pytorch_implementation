import torch
import torch.nn as nn
import torchvision.models as models
from deformable_conv2d.deformable_conv2d_wrapper import DeformableConv2DLayer


class BasicDeformableConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride_h, stride_w,
                 padding,
                 dilation_h=1, dilation_w=1,
                 num_groups=1,
                 deformable_groups=1,
                 im2col_step=1,
                 no_bias=True,
                 ):
        super(BasicDeformableConv2D, self).__init__()
        self.offset_generator = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * 2,
            kernel_size=kernel_size,
            padding=padding
        )
        self.mask_generator = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding
        )
        self.deformable_conv2d = DeformableConv2DLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride_h, stride_w,
            padding,
            dilation_h, dilation_w,
            num_groups,
            deformable_groups,
            im2col_step,
            no_bias
        )
        self.Sigmoid = nn.Sigmoid()
        # initialization
        nn.init.zeros_(self.offset_generator.weight)
        nn.init.zeros_(self.offset_generator.bias)
        nn.init.zeros_(self.mask_generator.weight)
        nn.init.zeros_(self.mask_generator.bias)

    def forward(self, x):
        offset = self.offset_generator(x)
        mask_origin = self.mask_generator(x)
        mask = self.Sigmoid(mask_origin)
        return self.deformable_conv2d(x, offset, mask)


class DeformableInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DeformableInceptionModule, self).__init__()

        # Deformable Layer
        self.deformable_conv_part_1 = BasicDeformableConv2D(in_channels, out_channels, 3, 1, 1, 1)
        self.deformable_conv_part_2 = BasicDeformableConv2D(in_channels, out_channels, 5, 1, 1, 2)
        self.deformable_conv_part_3 = BasicDeformableConv2D(in_channels, out_channels, 7, 1, 1, 3)

    def forward(self, x):
        # do the deformable convolution
        part_1 = self.deformable_conv_part_1(x)
        part_2 = self.deformable_conv_part_2(x)
        part_3 = self.deformable_conv_part_3(x)
        # concat
        output = torch.cat((part_1, part_2, part_3), dim=1)
        return output


class CONV2D1X1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CONV2D1X1, self).__init__()
        self.model = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
        nn.init.xavier_uniform_(self.model.weight, gain=1)

    def forward(self, x):
        return self.model(x)


class DMENet(nn.Module):
    def __init__(self):
        super(DMENet, self).__init__()
        # get front end
        self.front_end = nn.Sequential(*(list(list(models.vgg16(False).children())[0].children())[0:23]))
        # weight initialization
        self.front_end.apply(lambda m: nn.init.xavier_uniform_(m.weight, 1) if isinstance(m, nn.Conv2d) else None)
        # get back end
        self.back_end = nn.Sequential(
            DeformableInceptionModule(512, 256),
            CONV2D1X1(256 * 3, 256),
            DeformableInceptionModule(256, 128),
            CONV2D1X1(128 * 3, 128),
            DeformableInceptionModule(128, 64),
            CONV2D1X1(64 * 3, 1)
        )

    def forward(self, x):
        features = self.front_end(x)
        # if we need any process, code here
        out = self.back_end(features)
        return out
