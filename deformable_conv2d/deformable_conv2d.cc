#include <torch/extension.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
extern THCState *state;
#include <vector>
typedef std::vector<int> TShape;
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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

#ifndef FUNCTION_DECLARE
#define FUNCTION_DECLARE

    void deformable_im2col(cudaStream_t stream,
         const float* data_im, const float* data_offset, const float* data_mask,
         const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
         const TShape& pad, const TShape& stride, const TShape& dilation,
         const int32_t deformable_group, float* data_col);

    void deformable_col2im(cudaStream_t stream,
            const float* data_col, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int32_t deformable_group,
            float* grad_im);

    void deformable_col2im_coord(cudaStream_t stream,
            const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int32_t deformable_group,
            float* grad_offset, float* grad_mask);

    void setZero(cudaStream_t stream, int n, float* result_data);

    void setOne(cudaStream_t stream, int n, float* result_data);

    void pureAddTo(cudaStream_t stream, const int n, float* result_data, const float* right_data);

    void setNumAtIndex(cudaStream_t stream,  float num, int index, float* data);

    void SwapAxis(cudaStream_t stream, float* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);

#endif

at::Tensor deformable_conv2d_forward(
    at::Tensor input,
    at::Tensor filter,
    at::Tensor offset,
    at::Tensor mask,
    int stride_h,
    int stride_w,
    int num_groups,
    int deformable_groups,
    int im2col_step,
    bool no_bias,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
){
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(filter.type().is_cuda(), "filter must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int height = input.size(2);
    const int width = input.size(3);
    int kernel_h = filter.size(2);
    int kernel_w = filter.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int num_axes = 4;
    bool is_1x1_ = true;
    for (int32_t i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    int num_ = input.size(0);// batch size
    int channels_ = input.size(1);// number of input channels
    int group_ = num_groups;//
    int conv_out_channels_ = filter.size(0); // output channel nums
    int conv_in_channels_ = channels_; // input channel nums

    int kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    int conv_out_spatial_dim_ = height_out * width_out;
    int im2col_step_ = std::min(im2col_step, num_);

    int input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    int input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    int input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    int32_t M = conv_out_channels_ / group_; // filter的数量
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = kernel_dim_;

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());
    auto output = at::empty({num_, conv_out_channels_, height_out, width_out}, input.options());

    auto input_ptr = input.data<float>();
    auto weight_ptr = filter.data<float>();
    auto offset_ptr = offset.data<float>();
    auto mask_ptr = mask.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto output_ptr = output.data<float>();

    TShape input_shape;
    TShape filter_shape;
    TShape col_buffer_shape;
    TShape stride_shape;
    TShape dilation_shape;
    TShape padding_shape;

    input_shape.push_back(input.size(0));
    input_shape.push_back(input.size(1));
    input_shape.push_back(input.size(2));
    input_shape.push_back(input.size(3));
    filter_shape.push_back(filter.size(2));
    filter_shape.push_back(filter.size(3));
    col_buffer_shape.push_back(conv_in_channels_ * filter.size(2) * filter.size(3));
    col_buffer_shape.push_back(im2col_step_);
    col_buffer_shape.push_back(height_out);
    col_buffer_shape.push_back(width_out);
    stride_shape.push_back(stride_h);
    stride_shape.push_back(stride_w);
    dilation_shape.push_back(dilation_h);
    dilation_shape.push_back(dilation_w);
    padding_shape.push_back(pad_h);
    padding_shape.push_back(pad_w);

    for (int n = 0; n < num_ / im2col_step_; ++n) {
            deformable_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * im2col_step_ * input_dim_,
            offset_ptr + n * im2col_step_ * input_offset_dim_,
            mask_ptr + n * im2col_step_ * input_mask_dim_,
            input_shape,
            col_buffer_shape,
            filter_shape,
            padding_shape,
            stride_shape,
            dilation_shape,
            deformable_groups,
            col_buffer_ptr
            );
            auto output_instance_ptr = output_ptr + (n * group_ * M  * N);
            auto weight_ptr_ptr = &weight_ptr;
            auto col_buffer_ptr_ptr = &col_buffer_ptr;
            auto output_instance_ptr_ptr = &output_instance_ptr;
            THCudaBlas_SgemmBatched(state, 'n', 'n', M, N, K, 1.0f, (const float**)weight_ptr_ptr, M, (const float**)col_buffer_ptr_ptr, K, 1.0f, output_instance_ptr_ptr, M, group_);
//          SwapAxis<Device, T>(d, output_temp_4d_ptr, ToVector(TensorShape({num_ / im2col_step_, conv_out_channels_, im2col_step_, conv_out_spatial_dim_})), 1, 2);
    }
    return output;
}

std::vector<at::Tensor> deformable_conv2d_backward(
    at::Tensor input,
    at::Tensor filter,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor out_grad,
    int stride_h,
    int stride_w,
    int num_groups,
    int deformable_groups,
    int im2col_step,
    bool no_bias,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
){
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(filter.type().is_cuda(), "filter must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    AT_ASSERTM(out_grad.type().is_cuda(), "mask must be a CUDA tensor");


    const int height = input.size(2);
    const int width = input.size(3);
    int kernel_h = filter.size(2);
    int kernel_w = filter.size(3);

    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));
    const int32_t num_axes = 4;
    bool is_1x1_ = true;
    for (int32_t i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    int num_ = input.size(0);// batch size
    int channels_ = input.size(1);// number of input channels
    int group_ = num_groups;//
    int conv_out_channels_ = filter.size(0); // output channel nums
    int conv_in_channels_ = channels_; // input channel nums

    int kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    int conv_out_spatial_dim_ = height_out * width_out;
    int im2col_step_ = std::min(im2col_step, num_);

    int input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    int input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    int input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(filter);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    auto grad_weight_temp = at::zeros_like(filter);

    auto input_ptr = input.data<float>();
    auto weight_ptr = filter.data<float>();
    auto offset_ptr = offset.data<float>();
    auto mask_ptr = mask.data<float>();
    auto out_grad_ptr = out_grad.data<float>();
    auto grad_input_ptr = grad_input.data<float>();
    auto grad_weight_ptr = grad_weight.data<float>();
    auto grad_weight_temp_ptr = grad_weight_temp.data<float>();

    auto grad_offset_ptr = grad_offset.data<float>();
    auto grad_mask_ptr = grad_mask.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();

    int32_t M = kernel_dim_;
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = conv_out_channels_ / group_;

    TShape input_shape;
    TShape filter_shape;
    TShape col_buffer_shape;
    TShape stride_shape;
    TShape dilation_shape;
    TShape padding_shape;

    input_shape.push_back(input.size(0));
    input_shape.push_back(input.size(1));
    input_shape.push_back(input.size(2));
    input_shape.push_back(input.size(3));
    filter_shape.push_back(filter.size(2));
    filter_shape.push_back(filter.size(3));
    col_buffer_shape.push_back(conv_in_channels_ * filter.size(2) * filter.size(3));
    col_buffer_shape.push_back(im2col_step_);
    col_buffer_shape.push_back(height_out);
    col_buffer_shape.push_back(width_out);
    stride_shape.push_back(stride_h);
    stride_shape.push_back(stride_w);
    dilation_shape.push_back(dilation_h);
    dilation_shape.push_back(dilation_w);
    padding_shape.push_back(padding_h);
    padding_shape.push_back(padding_w);

    for(int n = 0;n < num_ / im2col_step_ ;++n){
        auto out_grad_instance_ptr = out_grad_ptr + n * group_ * K * N;
        THCudaBlas_SgemmBatched(state, 't', 'n', M, N, K, 1.0f, (const float**)(&weight_ptr), M, (const float**)(&out_grad_instance_ptr), N, 1.0f, &col_buffer_ptr, M, group_);
        deformable_col2im_coord(
                THCState_getCurrentStream(state),
                col_buffer_ptr,
                input_ptr + n * im2col_step_ * input_dim_,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                grad_offset_ptr + n * im2col_step_ * input_offset_dim_,
                grad_mask_ptr + n * im2col_step_ * input_mask_dim_);

        deformable_col2im(
                THCState_getCurrentStream(state),
                col_buffer_ptr,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                grad_input_ptr + n * im2col_step_ * input_dim_);

        deformable_im2col(
                THCState_getCurrentStream(state),
                input_ptr + n * im2col_step_ * input_dim_,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                col_buffer_ptr);
        if(0==n)
            THCudaBlas_SgemmBatched(state, 'n', 't', K, M, N, 1.0f, (const float**)(&out_grad_instance_ptr), K, (const float**)(&col_buffer_ptr), N, 1.0f, &grad_weight_ptr, K, group_);
        else{
            THCudaBlas_SgemmBatched(state, 'n', 't', K, M, N, 1.0f, (const float**)(&out_grad_instance_ptr), K, (const float**)(&col_buffer_ptr), N, 1.0f, &grad_weight_temp_ptr, K, group_);
            pureAddTo(THCState_getCurrentStream(state), K * M, grad_weight_ptr, grad_weight_temp_ptr);
        }

    }
    return {grad_input, grad_weight, grad_offset, grad_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deformable_conv2d_forward, "deformable_conv2d forward (CUDA)");
  m.def("backward", &deformable_conv2d_backward, "deformable_conv2d backward (CUDA)");
}