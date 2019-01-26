#include 'deformable_conv2d.h'

extern THCState *state;

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename DType>
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

    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = 4;
    num_spatial_axes_ = num_axes - first_spatial_axis; //表示的是空间坐标个数,比如说2维卷积里,就是2, 3维卷积里就是3, only implement for 3d convolution
    is_1x1_ = true; 
    for (int32_t i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    num_ = input.size(0);// batch size
    channels_ = input.size(1);// number of input channels
    group_ = num_groups;//
    conv_out_channels_ = filter.size(0); // output channel nums
    conv_in_channels_ = channels_; // input channel nums

    kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = height_out * width_out;
    im2col_step_ = std::min(im2col_step, num_);

    input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    int32_t M = conv_out_channels_ / group_; // filter的数量
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = kernel_dim_;

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    auto input_ptr = input.data<DType>();
    auto weight_ptr = filter.data<DType>();
    auto offset_ptr = offset.data<DType>();
    auto mask_ptr = mask.data<DType>();
    auto col_buffer_ptr = col_buffer.data<DType>();
    auto output_ptr = output.data<DType>();

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
            deformable_im2col<DType>(
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
            THCudaBlas_SgemmBatched(state, 'n', 'n', M, N, K, 1.0f, weight_ptr, M, col_buffer_ptr, K, 1.0f, output_instance_ptr, M, group_);
//          SwapAxis<Device, T>()(d, output_temp_4d_ptr, ToVector(TensorShape({num_ / im2col_step_, conv_out_channels_, im2col_step_, conv_out_spatial_dim_})), 1, 2);
    }
    return output;
}

template <typename DType>
std::vector<at::Tensor> deformable_conv2d_backaward(
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

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));

    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = 4;
    num_spatial_axes_ = num_axes - first_spatial_axis; //表示的是空间坐标个数,比如说2维卷积里,就是2, 3维卷积里就是3, only implement for 3d convolution
    is_1x1_ = true;
    for (int32_t i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    num_ = input.size(0);// batch size
    channels_ = input.size(1);// number of input channels
    group_ = num_groups;//
    conv_out_channels_ = filter.size(0); // output channel nums
    conv_in_channels_ = channels_; // input channel nums

    kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = height_out * width_out;
    im2col_step_ = std::min(im2col_step, num_);

    input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    auto input_ptr = input.data<DType>();
    auto weight_ptr = filter.data<DType>();
    auto offset_ptr = offset.data<DType>();
    auto mask_ptr = mask.data<DType>();
    auto out_grad_ptr = out_grad.data<DType>();
    auto grad_input_ptr = grad_input.data<DType>();
    auto grad_weight_ptr = grad_weight.data<DType>();
    auto grad_offset_ptr = grad_offset.data<DType>();
    auto grad_mask_ptr = grad_mask.data<DType>();
    auto col_buffer_ptr = col_buffer.data<DType>();

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
    padding_shape.push_back(pad_h);
    padding_shape.push_back(pad_w);

    for(int n = 0;n < num_ / im2col_step_ ;++n){
        auto out_grad_instance_ptr = out_grad_ptr + n * group_ * K * N;
        THCudaBlas_SgemmBatched(state, 't', 'n', m, n, k, 1.0f, weight_ptr, m, out_grad_instance_ptr, n, 1.0f, col_buffer_ptr, group_);
        deformable_col2im_coord<DType>(
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

        deformable_col2im<DType>(
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

        deformable_im2col<DType>(
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
        THCudaBlas_SgemmBatched(state, 't', 'n', m, n, k, 1.0f, weight_ptr, m, out_grad_instance_ptr, n, 1.0f, col_buffer_ptr, group_);
    }
    return { grad_input, grad_weight, grad_offset, grad_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deformable_conv2d_forward, "deformable_convolution forward (CUDA)");
  m.def("backward", &deformable_conv2d_backaward, "deformable_convolution backward (CUDA)");
}