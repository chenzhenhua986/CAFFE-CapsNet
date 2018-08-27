#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/capsule_mask_layer.hpp"

namespace caffe {
template <typename Dtype>
void CapsuleMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  class_num_ = this->layer_param_.capsule_mask_param().class_num();
  const int total = bottom[0]->count(1);
  capsule_dim_ = total / class_num_;
}

template <typename Dtype>
void CapsuleMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  v_.Reshape(bottom_shape);
  M_ = bottom[0]->count(0, 1);
  vector<int> max_shape;
  max_shape.push_back(M_);
  // max_shape.push_back(class_num);
  max_.Reshape(max_shape);
  top[0]->Reshape(bottom_shape);
}

template<typename Dtype>
Dtype cap_len(const Dtype* s, const int len, Dtype* v){
  caffe_copy(len, s, v);
  caffe_powx(len, v, Dtype(2), v);
  Dtype squared_norm = caffe_cpu_asum(len, v) + 1e-07;
  return sqrt(squared_norm);
}

template <typename Dtype>
void CapsuleMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(M_ * class_num_ * capsule_dim_, bottom_data, top_data);
  Dtype* v_data = v_.mutable_cpu_data();
  Dtype* max_data = max_.mutable_cpu_data();
  for (int i = 0; i < M_; ++i) {
    int max_index = 0;
    Dtype max_val = -FLT_MAX;
    for (int j = 0; j < class_num_; ++j) {
      Dtype len = cap_len(bottom_data + i * class_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
	v_data + i * class_num_ * capsule_dim_ + j * capsule_dim_);
      if(len > max_val){
	max_index = j;
      }
    }
    max_data[i] = max_index;
    for (int j = 0; j < class_num_; ++j) {
      if(j != max_index) {
	caffe_set(capsule_dim_, (Dtype)0., top_data + i * class_num_ * capsule_dim_ + j * capsule_dim_);	
      }
    }
  }
}


template <typename Dtype>
void CapsuleMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* max_data = max_.mutable_cpu_data();
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(M_ * class_num_ * capsule_dim_, top_diff, bottom_diff);
    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < class_num_; ++j) {
        if(j != max_data[i]) {
	  caffe_set(capsule_dim_, (Dtype)0., bottom_diff + i * class_num_ * capsule_dim_ + j * capsule_dim_);
        }
      }
    }	
  }
}


#ifdef CPU_ONLY
STUB_GPU(CapsuleMaskLayer);
#endif

INSTANTIATE_CLASS(CapsuleMaskLayer);
REGISTER_LAYER_CLASS(CapsuleMask);

}  // namespace caffe
