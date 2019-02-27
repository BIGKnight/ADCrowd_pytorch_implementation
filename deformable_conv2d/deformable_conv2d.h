// why we need to use the ifndef macro, because we need to assure the definition of class and struct only being defined once
#ifndef DEFORMABLE_CONV_2D_HEADER
#define DEFORMABLE_CONV_2D_HEADER
// A really strange thing took place. Once the header torch/extension.h was put in this header, the .cu which include current header can not be compiled by nvcc, it may some parts in the extension.h can not be compiled by nvcc
// according to the official document, the whole ATen library has been included by the torch/extension.h
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
typedef std::vector<int> TShape;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}

inline TShape SubVector(const TShape &shape, int start, int end) {
    TShape res;
    for(int i=start;i<end;i++){
        res.push_back(shape[i]);
    }
    return res;
}

// I think I find a big mistake that the function template can not be declared in this header and defined in .cu file and used in .cpp file, its incorrect.
// 在.h文件里声明,在.cpp文件里定义，然后在main函数里包含.h头文件，这样会报链接错误。这是因为函数模板要被实例化后才能成为真正的函数，
// 在使用函数模板的源文件中包含函数模板的头文件，如果该头文件中只有声明，没有定义，那编译器无法实例化该模板，最终导致链接错误。
// 我暂时想不出怎么解决这个问题, 貌似唯一的方式就是使用functor然后在.cu里实例化, 如果是两个cpp就可以直接include那个, include .cu肯定编译不过
// ImportError: /home/zzn/anaconda3/envs/pytorch/lib/python3.6/site-packages/deformable_conv2d-0.0.0-py3.6-linux-x86_64.egg/deformable_conv2d_gpu.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _Z17deformable_im2colIfEvP11CUstream_stPKT_S4_S4_RKSt6vectorIiSaIiEES9_S9_S9_S9_S9_iPS2_
// however, 我按照上面说的改了还是碰到了undefined symbol的问题, 然后我尝试了去掉deformable_im2col函数的定义, 并将其在.h中定义, 然后惊奇的发现报错重复定义, 但是此时.cu中的相应声明我早就
// 注释掉了, 然后我突然顿悟, 我发现在setup的时候, 是分别编译nvcc和g++的, 那么尽管有ifndef这个条件编译, .h这个文件依然被编译了2次, 那么自然在g++和并这两个的时候会报重复定义
// 所以, 我觉得之前undefined symbol是因为重复编译了头文件导致有多个函数声明. 因为之前tensorflow版本我是直接g++ -std=c++11 -shared -o deformable_conv2d.so deformable_conv2d.cc deformable_conv2d.cu.o 的, 所以不会出现这个问题
// 经测试, 两个相同名称的结构体是不允许存在的
    struct deformable_col2im{
        void operator()(cudaStream_t stream,
            const float* data_col, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int32_t deformable_group,
            float* grad_im);
        };

extern "C"{
    void deformable_im2col(cudaStream_t stream,
            const float* data_im, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride, const TShape& dilation,
            const int32_t deformable_group, float* data_col);
}


    struct deformable_col2im_coord{
        void operator()(cudaStream_t stream,
            const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int32_t deformable_group,
            float* grad_offset, float* grad_mask);
        };

    struct setZero{
        void operator()(cudaStream_t stream, int n, float* result_data);
    };

    struct setOne{
        void operator()(cudaStream_t stream, int n, float* result_data);
    };

    struct pureAddTo{
        void operator()(cudaStream_t stream, const int n, float* result_data, const float* right_data);
    };

    struct setNumAtIndex{
        void operator()(cudaStream_t stream,  float num, int index, float* data);
    };

    struct SwapAxis{
        void operator()(cudaStream_t stream, float* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);
    };


#endif
