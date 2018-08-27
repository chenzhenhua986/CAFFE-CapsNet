#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layers/capsule_len_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapsuleLenLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* lens = top[0]->mutable_gpu_data();
  Dtype* tmp = tmp_.mutable_gpu_data();
  caffe_gpu_powx(M_ * num_class_ * dim_, bottom_data, Dtype(2), tmp);
  //LOG(INFO) << "forward in capsule len layer ends: ";
  for(int i = 0; i < M_; ++i) {
    for(int j = 0; j < num_class_; ++j) {
      Dtype cap_len;
      caffe_gpu_asum(dim_, tmp + i * num_class_ * dim_ + j * dim_, &cap_len);
      caffe_gpu_set(1, cap_len, lens + i * num_class_ + j);
    }
  }
}  


template <typename Dtype>
void CapsuleLenLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate, for calculating accuracy only.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleLenLayer);
}  // namespace caffe
