from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='deformable_conv2d_cuda',
      ext_modules=[CUDAExtension('deformable_conv2d_gpu', ['deformable_conv2d_cuda.cu', 'deformable_conv2d.cc',]),],
      cmdclass={'build_ext': BuildExtension})

# it is the non-cuda setup script here
# setup(name='deformable_conv2d',
#       ext_modules=[CppExtension('deformable_conv2d', ['deformable_conv2d.cc'])],
#       cmdclass={'build_ext': BuildExtension})
# the equivalent code:
# setuptools.Extension(name='deformable_conv2d', sources=['deformable_conv2d.cc'],
# include_dirs=torch.utils.cpp_extension.include_paths(), language='c++')
