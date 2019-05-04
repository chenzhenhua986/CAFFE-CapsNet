#include <vector>
#include <cfloat>

#include "caffe/layers/dynamic_routing_layer.hpp"
//#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {


template <typename Dtype>
void DynamicRoutingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_output = this->layer_param_.cap_param().num_output();
  routing_num_ = this->layer_param_.dr_param().routing_num();
  input_capsule_num_ = this->layer_param_.dr_param().input_capsule_num();
  output_capsule_dim_ = this->layer_param_.dr_param().output_capsule_dim();
  output_capsule_num_ = this->layer_param_.dr_param().output_capsule_num();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(3);
    weight_shape[0] = 100;
    weight_shape[1] = output_capsule_num_;
    weight_shape[2] = input_capsule_num_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.capsule_routing_param().weight_filler()));
    //weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  /*
  vector<int> scale_shape;
  scale_shape.push_back(output_capsule_num_);
  scale_shape.push_back(input_capsule_num_);
  scale_.Reshape(scale_shape);
  b_.Reshape(scale_shape);
  */
  /*
  vector<int> v_shape;
  v_shape.push_back(output_capsule_num_);
  v_shape.push_back(output_capsule_dim_);
  v_.Reshape(v_shape);
  s_.Reshape(v_shape);
  */
}

template <typename Dtype>
void DynamicRoutingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  vector<int> top_shape(3);
  top_shape[0] = M_;
  top_shape[1] = output_capsule_num_;
  top_shape[2] = output_capsule_dim_;
  top[0]->Reshape(top_shape);
  v_.Reshape(top_shape);
  s_.Reshape(top_shape);

  vector<int> u_shape;
  u_shape.push_back(M_);
  u_shape.push_back(output_capsule_num_*output_capsule_dim_);
  u_shape.push_back(input_capsule_num_);
  u_.Reshape(u_shape);
  
  vector<int> scale_shape;
  scale_shape.push_back(M_);
  scale_shape.push_back(output_capsule_num_);
  scale_shape.push_back(input_capsule_num_);
  scale_.Reshape(scale_shape);
  b_.Reshape(scale_shape);
  /*
  vector<int> w_shape;
  w_shape.push_back(M_);
  w_shape.push_back(output_capsule_num_);
  w_shape.push_back(input_capsule_num_);
  weight_.Reshape(w_shape);
  */

  /*
  vector<int> weight_shape(3);
  weight_shape[0] = M_;
  weight_shape[1] = output_capsule_num_;
  weight_shape[2] = input_capsule_num_;
  */
}


template <typename Dtype>
void DynamicRoutingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  //const int routing_num = input_capsule_num_ * output_capsule_num_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* weight = weight_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* u_data = u_.mutable_cpu_data();
  Dtype* v_data = v_.mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();

  for(int ba = 0; ba < M_; ++ba) {
    // transpose u from 1152 * 160 to 160*1152
    caffe_cpu_transpose<Dtype>(input_capsule_num_, total_output, bottom_data + ba * input_capsule_num_ * total_output, u_data + ba * input_capsule_num_ * total_output);
    // update weight c=softmax(b) 
  }
  */
}


template <typename Dtype>
void DynamicRoutingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
/*  
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = weight_.mutable_cpu_data();
  Dtype* u_diff = u_.mutable_cpu_diff();
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data. top_diff: M_*10*16 bottom_data: M_*1152*160 w: 10*1152
    for(int ba = 0; ba < M_; ++ba) {
      for(int i = 0; i < output_capsule_num_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  output_capsule_dim_, input_capsule_num_, (Dtype)1.,  (Dtype)1., 
		top_diff + ba * total_output + i * output_capsule_dim_,  
			weight + i * input_capsule_num_, (Dtype)0.,
				u_diff + ba * input_capsule_num_ * total_output + i * input_capsule_num_ * output_capsule_dim_);
      }
      caffe_cpu_transpose<Dtype>(total_output, input_capsule_num_, u_diff + ba * input_capsule_num_ * total_output, bottom[0]->mutable_cpu_diff() + ba * input_capsule_num_ * total_output);
    }
  }
  */
}

#ifdef CPU_ONLY
STUB_GPU(DynamicRoutingLayer);
#endif

INSTANTIATE_CLASS(DynamicRoutingLayer);
REGISTER_LAYER_CLASS(DynamicRouting);

}
