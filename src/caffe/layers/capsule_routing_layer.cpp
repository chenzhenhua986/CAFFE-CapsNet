#include <vector>
#include <cfloat>

#include "caffe/layers/capsule_routing_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {


template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  input_capsule_num_ = this->layer_param_.capsule_routing_param().input_capsule_num();
  output_capsule_dim_ = this->layer_param_.capsule_routing_param().output_capsule_dim();
  output_capsule_num_ = this->layer_param_.capsule_routing_param().output_capsule_num();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = output_capsule_num_;
    weight_shape[1] = input_capsule_num_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_routing_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  // LOG(INFO) << "M_: "<<M_;
  vector<int> top_shape(3);
  top_shape[0] = M_;
  top_shape[1] = output_capsule_num_;
  top_shape[2] = output_capsule_dim_;
  top[0]->Reshape(top_shape);

  //u_ stores the transpose of inputs;
  vector<int> u_shape;
  u_shape.push_back(M_);
  u_shape.push_back(output_capsule_num_*output_capsule_dim_);
  u_shape.push_back(input_capsule_num_);
  u_.Reshape(u_shape);
}

template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int total_output = output_capsule_num_ * output_capsule_dim_;
  const int routing_num = input_capsule_num_ * output_capsule_num_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* u_data = u_.mutable_cpu_data();
  caffe_copy(M_ * input_capsule_num_ * total_output, bottom_data, u_data);
  for(int ba = 0; ba < M_; ++ba) {
    // transpose u from 1152 * 160 to 160*1152
    //caffe_cpu_transpose<Dtype>(input_capsule_num_, total_output, bottom_data + ba * input_capsule_num_ * total_output, u_data + ba * input_capsule_num_ * total_output);
    for(int i = 0; i < output_capsule_num_; ++i) {
      for(int j = 0; j < output_capsule_dim_; ++j) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (Dtype)1., (Dtype)1., input_capsule_num_, (Dtype)1., 
		u_data + ba * total_output * input_capsule_num_ + (i * output_capsule_dim_ + j)* input_capsule_num_, 
			weight + ba * routing_num + i * input_capsule_num_, (Dtype)0., 
				top_data + ba * total_output + i * output_capsule_dim_ + j);
      }
    }
  }
}

template <typename Dtype>
void CapsuleRoutingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, output_capsule_num_ * output_capsule_dim_, input_capsule_num_ * input_capsule_dim_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, input_capsule_num_ * input_capsule_dim_, output_capsule_num_ * output_capsule_dim_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleRoutingLayer);
#endif

INSTANTIATE_CLASS(CapsuleRoutingLayer);
REGISTER_LAYER_CLASS(CapsuleRouting);

}
