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

template <typename DType>
void deformable_im2col(cudaStream_t stream,
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col);


template <typename DType>
void deformable_col2im(cudaStream_t stream,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im);


template <typename DType>
void deformable_col2im_coord(cudaStream_t stream,
    const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_offset, DType* grad_mask);

template <typename DType>
void setZero(cudaStream_t stream, int n, DType* result_data);

template <typename DType>
void setOne(cudaStream_t stream, int n, DType* result_data);

template <typename DType>
void pureAddTo(cudaStream_t stream, const int n, DType* result_data, const DType* right_data);

template <typename DType>
void setNumAtIndex(cudaStream_t stream,  DType num, int index, DType* data);

template <typename DType>
void SwapAxis(cudaStream_t stream, DType* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);
#endif
