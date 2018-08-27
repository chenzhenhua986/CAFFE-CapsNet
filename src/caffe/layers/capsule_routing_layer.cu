#include <vector>
#include <math.h> 
#include <cfloat>

#include "caffe/layers/capsule_routing_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  //const int routing_num = input_capsule_num_ * output_capsule_num_;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  Dtype* u_data = u_.mutable_gpu_data();
  //LOG(INFO) << "forward in capsule routng layer starts: ";
  for(int ba = 0; ba < M_; ++ba) {
    // transpose u from 1152 * 160 to 160*1152
    caffe_gpu_transpose<Dtype>(input_capsule_num_, total_output, bottom_data + ba * input_capsule_num_ * total_output, u_data + ba * input_capsule_num_ * total_output);
    for(int i = 0; i < output_capsule_num_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_capsule_dim_, (Dtype)1., input_capsule_num_, (Dtype)1., 
		u_data + ba * total_output * input_capsule_num_ + i * output_capsule_dim_ * input_capsule_num_, 
			weight + i * input_capsule_num_, (Dtype)0., 
			top_data + ba * total_output + i * output_capsule_dim_);
    }
  }
  //LOG(INFO) << "forward in capsule routng layer ends: ";
}

template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* u_data = u_.mutable_gpu_data();
  Dtype* u_diff = u_.mutable_gpu_diff();
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  //LOG(INFO) << "backward in capsule routng layer starts: ";

  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight. top_diff: 10*16 bottom_data: 1152*160 w: 10*1152
    for(int ba = 0; ba < M_; ++ba) {
      caffe_gpu_transpose<Dtype>(input_capsule_num_, total_output, bottom_data + ba * input_capsule_num_ * total_output, u_data + ba * input_capsule_num_ * total_output);
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, (Dtype)1., input_capsule_num_, output_capsule_dim_, (Dtype)1., 
		top_diff + ba * total_output + i * output_capsule_dim_, 
			u_data + ba * total_output * input_capsule_num_ + i * input_capsule_num_ * output_capsule_dim_, (Dtype)1.0,  
			//u_data + ba * total_output * input_capsule_num_ + i * input_capsule_num_ * output_capsule_dim_, (Dtype)1.0 / (Dtype)M_,  
				this->blobs_[0]->mutable_gpu_diff() + i * input_capsule_num_);
      }
    }
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data. top_diff: M_*10*16 bottom_data: M_*1152*160 w: 10*1152
    for(int ba = 0; ba < M_; ++ba) {
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  output_capsule_dim_, input_capsule_num_, (Dtype)1.,  (Dtype)1., 
		top_diff + ba * total_output + i * output_capsule_dim_,  
			this->blobs_[0]->gpu_data() + i * input_capsule_num_, (Dtype)0.,
				//bottom[0]->mutable_gpu_diff() + ba * input_capsule_num_ * total_output + i * input_capsule_num_ * output_capsule_dim_);
				u_diff + ba * input_capsule_num_ * total_output + i * input_capsule_num_ * output_capsule_dim_);
      }
      caffe_gpu_transpose<Dtype>(total_output, input_capsule_num_, u_diff + ba * input_capsule_num_ * total_output, bottom[0]->mutable_gpu_diff() + ba * input_capsule_num_ * total_output);
    }
    /*
    for(int ba = 0; ba < M_; ++ba) {
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  input_capsule_num_, output_capsule_dim_, (Dtype)1.,  (Dtype)1., 
		this->blobs_[0]->gpu_data() + i * input_capsule_num_, 
			top_diff + ba * total_output + i * output_capsule_dim_, (Dtype)0.,  
			//top_diff + ba * total_output + i * output_capsule_dim_, (Dtype)1.0 / (Dtype)M_,  
				bottom[0]->mutable_gpu_diff() + ba * input_capsule_num_ * total_output + i * input_capsule_num_ * output_capsule_dim_);
      }
    }*/
  }
  //LOG(INFO) << "backward in capsule routng layer ends: ";
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleRoutingLayer);
} 
