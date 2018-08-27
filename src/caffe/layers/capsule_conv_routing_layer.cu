#include <vector>
#include <math.h> 
#include <cfloat>

#include "caffe/layers/capsule_conv_routing_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  vector<int> top_shape = top[0]->shape();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, output_capsule_dim_size_, (Dtype)1., input_capsule_num_ * kh_ * kw_, (Dtype)1., 
		bottom_data + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
			i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_, 
			weight + i * (kh_ * kw_ * input_capsule_num_), (Dtype)0., 
			//weight, (Dtype)0., 
			top_data + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_);
    }
  }	
}

template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight. top_diff: 32*6*6*16 (1152*16) bottom_data: 1152*(3*3*32)*16 w: 3*3*32
    for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (Dtype)1., (kh_ * kw_ * input_capsule_num_), output_capsule_dim_size_, (Dtype)1., 
		top_diff + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_, 
			bottom_data + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
				i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_, (Dtype)1.0,  
					this->blobs_[0]->mutable_gpu_diff() + i * (kh_ * kw_ * input_capsule_num_));
      }
    }
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data. top_diff: M_*32*6*6*16 bottom_data: M_*1152*288*16 w: 288*1
    for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  (kh_ * kw_ * input_capsule_num_), output_capsule_dim_size_, (Dtype)1.,  (Dtype)1., 
		this->blobs_[0]->gpu_data() + i * (kh_ * kw_ * input_capsule_num_),
			top_diff + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_, (Dtype)0., 
				bottom[0]->mutable_gpu_diff() + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
					i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleConvRoutingLayer);
} 
