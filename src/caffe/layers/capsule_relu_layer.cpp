#include <algorithm>
#include <vector>

#include "caffe/layers/capsule_relu_layer.hpp"

namespace caffe {
template <typename Dtype>
void CapsuleReluLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  dim_ = this->layer_param_.capsule_relu_param().dim();
  thre_ = this->layer_param_.capsule_relu_param().thre();
}


template <typename Dtype>
void CapsuleReluLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
  tmp_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void CapsuleReluLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = count / dim_;
  Dtype* tmp = tmp_.mutable_cpu_data();
  caffe_powx(count, bottom_data, Dtype(2), tmp);
  int relued = 0;
  for(int i = 0; i < num; ++i) {
    Dtype len = std::sqrt(caffe_cpu_asum(dim_, tmp + i * dim_) + 1e-07);
    if (len > thre_) {
      caffe_set(dim_, (Dtype)0.0, top_data + i * dim_);
      relued++;
    }
  }
  LOG(INFO) << "saturated rate: "<<float(relued) / float(num) * 100 <<"%";
}

template <typename Dtype>
void CapsuleReluLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = count / dim_;
    Dtype* tmp = tmp_.mutable_cpu_data();
    caffe_powx(count, bottom_data, Dtype(2), tmp);
    for(int i = 0; i < num; ++i) {
      Dtype len = std::sqrt(caffe_cpu_asum(dim_, tmp + i * dim_) + 1e-07);
      if (len <= thre_) {
        caffe_copy(dim_, top_diff + i * dim_, bottom_diff + i * dim_);
      }
    }
  }
}


INSTANTIATE_CLASS(CapsuleReluLayer);
REGISTER_LAYER_CLASS(CapsuleRelu);

}  // namespace caffe
