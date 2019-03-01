# def test_kwargs(first, *args, **kwargs):
#     print(kwargs.get("k1"))
#     for k, v in kwargs.items():
#         print('Optional argument %s (*kwargs): %s' % (k, v))
#
# test_kwargs(1, 2, 3, 4, k1=5, k2=6)
import torch
from deformable_conv2d.deformable_conv2d_wrapper import DeformableConv2DLayer
import torch.cuda as torch_cuda

assert torch_cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU


# NCHW
x = torch.ones(1, 3, 5, 5, dtype=torch.float32, device=cuda_device)
filter = torch.ones(3, 3, 3, 3, dtype=torch.float32, device=cuda_device)
offset = torch.ones(1, 18, 5, 5, dtype=torch.float32, device=cuda_device)
mask = torch.ones(1, 9, 5, 5, dtype=torch.float32, device=cuda_device)
DeformableLayer = DeformableConv2DLayer(1, 1, 1, 1).to(cuda_device)
y = DeformableLayer.forward(x, filter, offset, mask)
# print(y.cpu())
