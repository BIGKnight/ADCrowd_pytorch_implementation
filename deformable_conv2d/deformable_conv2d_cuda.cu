#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <vector>
#include 'deformable_conv2d.h'
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

typedef std::vector<int> TShape;
namespace{
template<typename DType>
__device__ DType dmcn_im2col_bilinear(
    const DType* bottom_data,
    const int data_width,
    const int height,
    const int width,
    DType h,
    DType w){

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  DType v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  DType v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  DType v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;

}

template<typename DType>
__device__ DType dmcn_get_gradient_weight(
    DType argmax_h, // offset h
    DType argmax_w, // offset w
    const int h,  const int w, // coordinate
    const int height,  const int width){

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  DType weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename DType>
__device__ DType dmcn_get_coordinate_weight(
    DType argmax_h,
    DType argmax_w,
    const int height,
    const int width,
    const DType* im_data,
    const int data_width,
    const int bp_dir
    ) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  DType weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename DType>
__global__ void SwapAxisKernel(
    const int n,
    const int cuda_mem_size, const int min_unit_size,
    DType* input_data,
    const int dim_num,
    const int axis_x_dims, const int axis_y_dims,
    const int axis_x, const int axis_y){
    CUDA_KERNEL_LOOP(index, n){
//        size_t size = cuda_mem_size * sizeof(DType);
        DType *device_data = NULL;

        device_data = new DType[cuda_mem_size];

//        cudaMalloc((void**)&device_data, size);
        DType* input_data_ptr = input_data + index * cuda_mem_size;
        for(int j =0;j<axis_y_dims;j++){
            for(int i=0;i<axis_x_dims;i++){
                DType* temp_ptr = input_data_ptr + (i * axis_x_dims + j) * min_unit_size;
//                cudaMemcpy(device_data + (j * axis_y_dims + i) * min_unit_size, temp_ptr, sizeof(DType)*min_unit_size, cudaMemcpyHostToDevice);
                DType* device_data_temp_ptr = device_data +  (j * axis_y_dims + i) * min_unit_size;
                for(int k = 0;k<min_unit_size;k++){
                    *(device_data_temp_ptr + k) = *(temp_ptr + k);
                }
            }
        }
//        cudaMemcpy(input_data_ptr, device_data, size, cudaMemcpyDeviceToHost);
        for(int i =0;i<cuda_mem_size;i++)
            *(input_data_ptr + i) = *(device_data + i);
    }
}

template <typename DType>
__global__ void DeformableConv2DIm2ColKernel(
    const int n,
    const DType* data_im,
    const DType* data_offset,
    const DType* data_mask,

    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,

    const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col,
    DType* data_col){
    CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    DType* data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const DType* data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const DType* data_mask_ptr = data_mask + (b_col *  deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        DType val = static_cast<DType>(0);
        const DType h_im = h_in + i * dilation_h + offset_h;
        const DType w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename DType>
__global__ void DeformableConv2DCol2ImKernel(
    const int n,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    DType* grad_im){
    CUDA_KERNEL_LOOP(index, n){
        const int j = (index / width_col / height_col / batch_size) % kernel_w;
        const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
        const int deformable_group_index = c / channel_per_deformable_group;
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int b = (index / width_col / height_col) % batch_size;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;
        const DType* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        const DType* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;
        const DType cur_top_grad = data_col[index] * mask;
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;
        for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1
            ) {
                int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
                DType weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
                atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}

template <typename DType>
__global__ void DeformableConv2DCol2ImCoordGPUKernel(
  const int n,
  const DType* data_col, const DType* data_im,
  const DType* data_offset, const DType* data_mask,
  const int channels, const int height, const int width, // 输入的C, H, W
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int batch_size, const int offset_channels, const int deformable_group,
  const int height_col, const int width_col,
  DType* grad_offset, DType* grad_mask) {
  CUDA_KERNEL_LOOP(index, n){
    DType val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const DType* data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const DType* data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const DType* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const DType* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;
    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;
      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      const DType mask = data_mask_ptr[data_mask_hw_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const DType weight = dmcn_get_coordinate_weight(
        inv_h, inv_w,
        height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val  += weight * data_col_ptr[col_pos] * mask;
      cnt  += 1;
    }

    grad_offset[index] = val;
    if (offset_c % 2 == 0){
            grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
        }
    }
}

template <typename DType>
__global__ void pureAddToKernel(const int n, DType* result_data, const DType* right_data){
      CUDA_KERNEL_LOOP(index, n) {
          CudaAtomicAdd(result_data+index, right_data[index]);
      }

    }

template <typename DType>
__global__ void setZeroKernel(const int n, DType* result_data){
         CUDA_KERNEL_LOOP(index, n){
          *(result_data + index) = DType(0);
      }

    }

template <typename DType>
__global__ void setOneKernel(const int n, DType* result_data){
        CUDA_KERNEL_LOOP(index, n){
            *(result_data + index) = DType(1);
        }
    }

template <typename DType>
__global__ void setNumAtIndexKernel(DType num, int index, DType* data){
        *(data + index) = num;
}

}

template <typename DType>
void deformable_im2col(cudaStream_t stream,
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col) {
        const int channel_per_deformable_group = channels / deformable_group;
        const int num_kernels = channels * batch_size * height_col * width_col;
        int  num_spatial_axes = kernel_shape.size();
        int  channel_per_deformable_group = im_shape[1] / deformable_group;
        int  num_kernels = im_shape[1] * ProdShape(col_shape, 1, col_shape.size());
        switch (num_spatial_axes) {
        case 2:
        DeformableConv2DIm2ColKernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            data_offset,
            data_mask,
            im_shape[2], im_shape[3],
            kernel_shape[0], kernel_shape[1],
            pad[0], pad[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            channel_per_deformable_group,
            col_shape[1], im_shape[1],
            deformable_group,
            col_shape[2], col_shape[3],
            data_col);
            break;
            default:
                cudaError_t err = cudaGetLastError();
                printf("error in DeformableConv2DIm2ColKernel: %s\n", cudaGetErrorString(err));
            }

}

template <typename DType>
void deformable_col2im(cudaStream_t stream,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im){
        int  num_spatial_axes = kernel_shape.size();
        int  im_size = ProdShape(im_shape, 1, im_shape.size());
        int  channel_per_deformable_group = im_shape[1] / deformable_group;
        int  num_kernels = ProdShape(col_shape, 0, col_shape.size());
          switch (num_spatial_axes) {
          case 2:
                DeformableConv2DCol2ImKernel<DType>
                <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                num_kernels, data_col, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
                kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
                dilation[0], dilation[1], channel_per_deformable_group,
                col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im);
            break;
          default:
            cudaError_t err = cudaGetLastError();
            printf("error in DeformableConv2DIm2ColKernel: %s\n", cudaGetErrorString(err));
          }

}

template <typename DType>
void deformable_col2im_coord(cudaStream_t stream,
    const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_offset, DType* grad_mask) {
      int  num_spatial_axes = kernel_shape.size();
      int  num_kernels = col_shape[1] * col_shape[2] * col_shape[3] * 2 * kernel_shape[0] * kernel_shape[1] * deformable_group;
      int  channel_per_deformable_group = col_shape[0] / deformable_group;
      switch (num_spatial_axes) {
      case 2:
        DeformableConv2DCol2ImCoordGPUKernel<DType>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, data_col, data_im, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
            kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
            dilation[0], dilation[1], channel_per_deformable_group,
            col_shape[1], 2 * kernel_shape[0] * kernel_shape[1] * deformable_group, deformable_group, col_shape[2], col_shape[3],
            grad_offset, grad_mask);
        break;
      default:
            cudaError_t err = cudaGetLastError();
            printf("error in DeformableConv2DCol2ImCoordGPUKernel: %s\n", cudaGetErrorString(err));
    }
}

template <typename DType>
void SwapAxis(cudaStream_t stream, DType* input_data, const TShape& origin_shape, const int axis_x, const int axis_y){
    return;
}

template <typename DType>
void setZero(cudaStream_t stream, int n, DType* result_data){
    setZeroKernel<DType> <<< GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream >>>(n, result_data);
}

template <typename DType>
void setOne(cudaStream_t stream, int n, DType* result_data){
    setOneKernel<DType> <<< GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream >>>(n, result_data);
}

template <typename DType>
void pureAddTo(cudaStream_t stream, const int n, DType* result_data, const DType* right_data){
    pureAddToKernel<DType> <<< GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream  >>>(n, result_data, right_data);
}

template <typename DType>
void setNumAtIndex(cudaStream_t stream,  DType num, int index, DType* data){
    setNumAtIndexKernel<DType> <<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream >>>(num, index, data);
}
