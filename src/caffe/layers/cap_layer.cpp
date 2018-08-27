#include <vector>
#include <cfloat>

#include "caffe/layers/cap_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {


template <typename Dtype>
void CapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_output = this->layer_param_.cap_param().num_output();
  input_capsule_dim_ = this->layer_param_.cap_param().input_capsule_dim();
  output_capsule_dim_ = this->layer_param_.cap_param().output_capsule_dim();
  output_capsule_num_ = this->layer_param_.cap_param().output_capsule_num();
  bias_term_ = this->layer_param_.cap_param().bias_term();
  //bias_term_ = 0;
  //const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.cap_param().axis());
  const int axis = 1;
  const int total_dim = bottom[0]->count(axis);
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
        this->layer_param_.cap_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, output_capsule_num_ * output_capsule_dim_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.cap_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  M_ = bottom[0]->count(0, 1);
  //const int total_input = input_capsule_num_*input_capsule_dim_;
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  vector<int> u_shape;
  u_shape.push_back(M_);
  u_shape.push_back(input_capsule_num_);
  u_shape.push_back(total_output);
  u_.Reshape(u_shape);
  vector<int> u_T_shape;
  u_T_shape.push_back(M_);
  u_T_shape.push_back(total_output);
  u_T_shape.push_back(input_capsule_num_);
  u_T_.Reshape(u_T_shape);
  //LOG(INFO) << "u_ reshape: ";
  vector<int> b_shape;
  b_shape.push_back(M_);
  b_shape.push_back(output_capsule_num_);
  b_shape.push_back(input_capsule_num_);
  b_.Reshape(b_shape);
  c_.Reshape(b_shape);
  vector<int> s_shape;
  s_shape.push_back(M_);
  s_shape.push_back(output_capsule_num_);
  s_shape.push_back(output_capsule_dim_);
  s_.Reshape(s_shape);
  v_.Reshape(s_shape);
  top[0]->Reshape(s_shape);
  vector<int> tmp;
  tmp.push_back(M_);
  tmp.push_back(output_capsule_num_);
  scale_.Reshape(tmp);

  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void CapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}


template <typename Dtype>
void CapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(CapLayer);
#endif

INSTANTIATE_CLASS(CapLayer);
REGISTER_LAYER_CLASS(Cap);

}
