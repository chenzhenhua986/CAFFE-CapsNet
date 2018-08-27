#include <vector>
#include <math.h> 
#include <cfloat>

//#include "thrust/device_vector.h"

#include "caffe/layers/cap_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
    //printf("data : %f", data[index]);
  }
}

template <typename Dtype>
void CapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int total_input = input_capsule_num_*input_capsule_dim_;
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  const int routing_num = input_capsule_num_ * output_capsule_num_;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  
  // Calculate u_.
  // Shape of input: input_capsule_num * input_capsule_dim, namely, 1152*8 
  // Shape of u_: input_capsule_num * (output_capsule_num * output_capsule_dim), namely, 1152*(16*10)
  // Shape of weights: input_capsule_num * input_capsule_dim * (output_capsule_num * output_capsule_dim), namely, 1152*8*10*16
  Dtype* u_data = u_.mutable_gpu_data();
  Dtype* u_T_data = u_T_.mutable_gpu_data();
  for(int i = 0; i < M_; ++i) {
    for(int j = 0; j < input_capsule_num_; ++j) {
      // 1*8 multiply 8*(16*10) = 1*(16*10)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, total_output, input_capsule_dim_, (Dtype)1., 
	//bottom_data_copy + i * total_input + j * input_capsule_dim_, 
	bottom_data + i * total_input + j * input_capsule_dim_, 
		weight + j * input_capsule_dim_ * total_output, (Dtype)0., 
			u_data + i * input_capsule_num_ * total_output + j * total_output);
    }
  }

  // Dynamic routing
  Dtype* c_data = c_.mutable_gpu_data();
  Dtype* s_data = s_.mutable_gpu_data();
  Dtype* v_data = v_.mutable_gpu_data();
  Dtype* b_data = b_.mutable_gpu_data();
 
  //No Routing
  /*
  caffe_gpu_set(M_ * routing_num, (Dtype)1.0, c_data);
    for(int ba = 0; ba < M_; ++ba) {
      // transpose u from 1152 * 160 to 160*1152
      caffe_gpu_transpose<Dtype>(input_capsule_num_, total_output, u_data + ba * input_capsule_num_ * total_output, u_T_data + ba * input_capsule_num_ * total_output);
      // Calculate s 10 * 1152 \times (1152*160)^T = (10*1152) * (10*16*1152) = 10*16
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_capsule_dim_, (Dtype)1., input_capsule_num_, (Dtype)1., 
		u_T_data + ba * total_output * input_capsule_num_ + i * output_capsule_dim_ * input_capsule_num_, 
			c_data + ba * routing_num + i * input_capsule_num_, (Dtype)0., 
				s_data + ba * total_output + i * output_capsule_dim_);
      }
    }
  */
  int routing_times = 3;
  //for(int r = 0; r < routing_times && phase_ == TRAIN; ++r) {
  for(int r = 0; r < routing_times; ++r) {
    caffe_gpu_memcpy(M_ * routing_num * sizeof(Dtype), b_data, c_data);
    kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(M_ * input_capsule_num_), CAFFE_CUDA_NUM_THREADS>>>(M_, output_capsule_num_, input_capsule_num_, c_data, scale_data);
    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(M_ * routing_num), CAFFE_CUDA_NUM_THREADS>>>(M_ * routing_num, M_, output_capsule_num_, input_capsule_num_, scale_data, c_data);
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(M_ * routing_num), CAFFE_CUDA_NUM_THREADS>>>(M_ * routing_num, c_data, c_data);
    kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(M_ * input_capsule_num_), CAFFE_CUDA_NUM_THREADS>>>(M_, output_capsule_num_, input_capsule_num_, c_data, scale_data);
    kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(M_ * routing_num), CAFFE_CUDA_NUM_THREADS>>>(M_ * routing_num, M_, output_capsule_num_, input_capsule_num_, scale_data, c_data);

    for(int ba = 0; ba < M_; ++ba) {
      // transpose u from 1152 * 160 to 160*1152
      caffe_gpu_transpose<Dtype>(input_capsule_num_, total_output, u_data + ba * input_capsule_num_ * total_output, u_T_data + ba * input_capsule_num_ * total_output);
      // Calculate s 10 * 1152 \times (1152*160)^T = (10*1152) * (10*16*1152) = 10*16
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_capsule_dim_, (Dtype)1., input_capsule_num_, (Dtype)1., 
		u_T_data + ba * total_output * input_capsule_num_ + i * output_capsule_dim_ * input_capsule_num_, 
			c_data + ba * routing_num + i * input_capsule_num_, (Dtype)0., 
				s_data + ba * total_output + i * output_capsule_dim_);
      }

      // Squash
      //caffe_gpu_memcpy(total_output, s_data + ba * total_output, v_data + ba * total_output);
      Dtype squared_norm;
      for (int j = 0; j < output_capsule_num_; ++j) {
        caffe_gpu_powx(output_capsule_dim_, s_data + ba * total_output + j * output_capsule_dim_, Dtype(2), v_data + ba * total_output + j * output_capsule_dim_);
        caffe_gpu_asum(output_capsule_dim_, v_data + ba * total_output + j * output_capsule_dim_, &squared_norm);
        //cap_len(s_data + ba * total_output + j * output_capsule_dim_, output_capsule_dim_, &len, v_data + ba * total_output + j * output_capsule_dim_);
	Dtype tmp = squared_norm / (1.0 + squared_norm) / sqrt(squared_norm + 1e-7);
	caffe_gpu_scale(output_capsule_dim_, tmp, s_data + ba * total_output + j * output_capsule_dim_, v_data + ba * total_output + j * output_capsule_dim_);
        //LOG(INFO) << "len value: "<<squared_norm;
      }
      // Re-calculate b. u: 1152*10*(16), v: 10*16, b: 10*1152
      for(int i = 0; i < output_capsule_num_; ++i) {
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1., input_capsule_num_, output_capsule_dim_, (Dtype)1., 
		v_data + ba * total_output + i * output_capsule_dim_,
			u_T_data + ba * input_capsule_num_ * total_output + i * input_capsule_num_ * output_capsule_dim_, (Dtype)1., 
				b_data + ba * routing_num + i * input_capsule_num_);
      }
    }
  }
        
  //caffe_gpu_add_scalar<Dtype>(M_*total_output, 1e-1, v_data);
  Dtype* top_data = top[0]->mutable_gpu_data();
  //caffe_gpu_memcpy(M_ * total_output * sizeof(Dtype), v_data, top_data);
  caffe_gpu_memcpy(M_ * total_output * sizeof(Dtype), s_data, top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, total_output, 1, (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}


template <typename Dtype>
void CapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    for(int i = 0; i < M_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, input_capsule_num_ * input_capsule_dim_, output_capsule_num_ * output_capsule_dim_, 1,
		(Dtype)1., bottom_data + i * input_capsule_num_ * input_capsule_dim_, top_diff + i * output_capsule_num_ * output_capsule_dim_, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }

    /*
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, input_capsule_num_ * input_capsule_dim_, output_capsule_num_ * output_capsule_dim_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    */
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    //caffe_gpu_gemv<Dtype>(CblasTrans, M_, output_capsule_num_ * output_capsule_dim_, (Dtype)1., top_diff, bias_multiplier_.gpu_data(), (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, output_capsule_num_ * output_capsule_dim_, (Dtype)1., top_diff, bias_multiplier_.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    for(int i = 0; i < M_; ++i) {
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, input_capsule_num_ * input_capsule_dim_, output_capsule_num_ * output_capsule_dim_,
		(Dtype)1., top_diff + i * output_capsule_num_ * output_capsule_dim_, this->blobs_[0]->gpu_data(), (Dtype)0., bottom[0]->mutable_gpu_diff() + i * input_capsule_num_ * input_capsule_dim_);
    }
    /*
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, input_capsule_num_ * input_capsule_dim_, output_capsule_num_ * output_capsule_dim_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    */
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapLayer);

} 
