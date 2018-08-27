#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layers/capsule_len_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapsuleLenLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_class_ = this->layer_param_.capsule_len_param().num_class();
}

template <typename Dtype>
void CapsuleLenLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  dim_ = bottom[0]->count(1) / num_class_;
  vector<int> cap_lens_shape;
  cap_lens_shape.push_back(M_);
  cap_lens_shape.push_back(num_class_);
  tmp_.Reshape(bottom[0]->shape());
  top[0]->Reshape(cap_lens_shape);
}

template <typename Dtype>
void CapsuleLenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* lens = top[0]->mutable_cpu_data();
  Dtype* tmp = tmp_.mutable_cpu_data();
  caffe_powx(M_ * num_class_ * dim_, bottom_data, Dtype(2), tmp);
  for(int i = 0; i < M_; ++i) {
    for(int j = 0; j < num_class_; ++j) {
      lens[i * num_class_ + j] = std::sqrt(caffe_cpu_asum(dim_, tmp + i * num_class_ * dim_ + j * dim_) + 1e-07);
    }
  }
}  


template <typename Dtype>
void CapsuleLenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate, for calculating accuracy only.";
  }
}

INSTANTIATE_CLASS(CapsuleLenLayer);
REGISTER_LAYER_CLASS(CapsuleLen);

}  // namespace caffe
