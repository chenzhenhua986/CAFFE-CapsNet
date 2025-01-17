#include <vector>
#include <cfloat>

#include "caffe/layers/capsule_transform_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {

template <typename Dtype>
void CapsuleTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  input_capsule_dim_ = this->layer_param_.capsule_transform_param().input_capsule_dim();
  output_capsule_dim_ = this->layer_param_.capsule_transform_param().output_capsule_dim();
  output_capsule_num_ = this->layer_param_.capsule_transform_param().output_capsule_num();
  bias_term_ = this->layer_param_.capsule_transform_param().bias_term();
  const int total_dim = bottom[0]->count(1);
  input_capsule_num_ = total_dim / input_capsule_dim_;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(3);
    weight_shape[0] = input_capsule_num_;
    weight_shape[1] = input_capsule_dim_;
    weight_shape[2] = output_capsule_num_ * output_capsule_dim_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_transform_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      //vector<int> bias_shape(1, output_capsule_num_ * output_capsule_dim_);
      vector<int> bias_shape(1, input_capsule_num_ * output_capsule_num_ * output_capsule_dim_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.capsule_transform_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  //vector<int> bottom_shape = bottom[0]->shape();
  //bottom_shape.resize(3);
  //bottom_shape[1] = bottom[0]->count(1) / input_capsule_dim_;
  //bottom_shape[2] = input_capsule_dim_;
  //bottom[0]->Reshape(bottom_shape);
  vector<int> top_capsule_shape(3);
  top_capsule_shape[0] = M_;
  top_capsule_shape[1] = input_capsule_num_;
  top_capsule_shape[2] = output_capsule_num_ * output_capsule_dim_;
  top[0]->Reshape(top_capsule_shape);

  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void CapsuleTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
/*
  const int total_input = input_capsule_num_*input_capsule_dim_;
  const int total_output = input_capsule_num_ * output_capsule_num_ * output_capsule_dim_;
  //const int routing_num = input_capsule_num_ * output_capsule_num_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for(int i = 0; i < M_; ++i) {
    for(int j = 0; j < input_capsule_num_; ++j) {
      // 1*8 multiply 8*(16*10) = 1*(16*10)
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, total_output, input_capsule_dim_, (Dtype)1., 
	bottom_data + i * total_input + j * input_capsule_dim_, 
		weight + j * input_capsule_dim_ * total_output, (Dtype)0., 
			top_data + i * input_capsule_num_ * total_output + j * total_output);
    }
  }

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, total_output, 1, (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
*/
}


template <typename Dtype>
void CapsuleTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
/*

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, input_capsule_num_*output_capsule_num_ * output_capsule_dim_, input_capsule_num_ * input_capsule_dim_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, input_capsule_num_ * output_capsule_num_*output_capsule_dim_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, input_capsule_num_ * input_capsule_dim_, input_capsule_num_ * output_capsule_num_ * output_capsule_dim_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
*/
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleTransformLayer);
#endif

INSTANTIATE_CLASS(CapsuleTransformLayer);
REGISTER_LAYER_CLASS(CapsuleTransform);

}
