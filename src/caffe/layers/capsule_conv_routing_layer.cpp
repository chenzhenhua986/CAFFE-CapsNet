#include <vector>

#include "caffe/layers/capsule_conv_routing_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {


template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  stride_ = this->layer_param_.capsule_conv_routing_param().stride();
  kh_ = this->layer_param_.capsule_conv_routing_param().kh();
  kw_ = this->layer_param_.capsule_conv_routing_param().kw();
  input_capsule_num_ = this->layer_param_.capsule_conv_routing_param().input_capsule_num();
  output_capsule_num_ = this->layer_param_.capsule_conv_routing_param().output_capsule_num();
  input_h_ = this->layer_param_.capsule_conv_routing_param().input_h();
  input_w_ = this->layer_param_.capsule_conv_routing_param().input_w();
  output_h_ = (input_h_ - kh_) / stride_ + 1;
  output_w_ = (input_w_ - kw_) / stride_ + 1;
  const BlobShape& output_capsule_shape = this->layer_param_.capsule_conv_routing_param().output_capsule_shape();
  output_capsule_dim_size_ = output_capsule_shape.dim(0) * output_capsule_shape.dim(1);
  //LOG(INFO) << "out cap dim size: "<<output_capsule_dim_size_;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(2);
    // weight_shape[0] = 1;
    weight_shape[0] = output_h_ * output_w_ * output_capsule_num_;
    weight_shape[1] = kh_ * kw_ * input_capsule_num_;
    //vector<int> weight_shape(1);
    //weight_shape[0] = kh_ * kw_ * input_capsule_num_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_conv_routing_param().weight_filler()));
    //LOG(INFO) << "gpu diff: "<<this->blobs_[0]->gpu_diff();
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  vector<int> top_shape(3);
  top_shape[0] = M_;
  top_shape[1] = output_capsule_num_ * output_h_ * output_w_;
  top_shape[2] = output_capsule_dim_size_;
  //LOG(INFO) << "gpu data: "<<top[0]->gpu_data();
  //LOG(INFO) << "gpu diff: "<<top[0]->gpu_diff();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  vector<int> top_shape = top[0]->shape();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, output_capsule_dim_size_, (Dtype)1., input_capsule_num_ * kh_ * kw_, (Dtype)1., 
		bottom_data + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
			i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_, 
			weight + i * (kh_ * kw_ * input_capsule_num_), (Dtype)0., 
			top_data + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_);
    }
  }
}

template <typename Dtype>
void CapsuleConvRoutingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight. top_diff: 32*6*6*16 (1152*16) bottom_data: 1152*(3*3*32)*16 w: 3*3*32
    for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (Dtype)1., (kh_ * kw_ * input_capsule_num_), output_capsule_dim_size_, (Dtype)1., 
		top_diff + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_, 
			bottom_data + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
				i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_, (Dtype)1.0,  
					this->blobs_[0]->mutable_cpu_diff() + i * (kh_ * kw_ * input_capsule_num_));
      }
    }
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data. top_diff: M_*32*6*6*16 bottom_data: M_*1152*288*16 w: 288*1
    for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_capsule_num_ * output_w_ * output_h_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  (kh_ * kw_ * input_capsule_num_), output_capsule_dim_size_, (Dtype)1.,  (Dtype)1., 
		this->blobs_[0]->cpu_data() + i * (kh_ * kw_ * input_capsule_num_),
			top_diff + b * (output_capsule_num_ * output_h_ * output_w_) * output_capsule_dim_size_ + i * output_capsule_dim_size_, (Dtype)0., 
				bottom[0]->mutable_cpu_diff() + b * (output_capsule_num_ * output_h_ * output_w_) * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_ + 
					i * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim_size_);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleConvRoutingLayer);
#endif

INSTANTIATE_CLASS(CapsuleConvRoutingLayer);
REGISTER_LAYER_CLASS(CapsuleConvRouting);

}
