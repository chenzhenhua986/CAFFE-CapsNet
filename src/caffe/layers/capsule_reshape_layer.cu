#include <algorithm>
#include <vector>

#include "caffe/layers/capsule_reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int feature_size = capsule_num_ / group_num_; // 1152 / 32 = 36 = 6*6
  for(int ba = 0; ba < M_; ++ba) {
    for(int i = 0; i < group_num_; ++i) {
      caffe_gpu_transpose<Dtype>(capsule_dim_, feature_size, 
	bottom_data + ba * capsule_num_ * capsule_dim_ + i * feature_size * capsule_dim_, 
		top_data + ba * capsule_num_ * capsule_dim_ + i * feature_size * capsule_dim_);
    }
  }
}

template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //LOG(INFO) << "in capsule reshape layer starts: "<< propagate_down[0];
    const Dtype* top_diff = top[0]->gpu_diff();
    const int feature_size = capsule_num_ / group_num_; // 1152 / 32 = 36 = 6*6
    for(int ba = 0; ba < M_; ++ba) {
      for(int i = 0; i < group_num_; ++i) {
        caffe_gpu_transpose<Dtype>(feature_size, capsule_dim_, 
	    top_diff + ba * capsule_num_ * capsule_dim_ + i * feature_size * capsule_dim_, 
		bottom[0]->mutable_gpu_diff() + ba * capsule_num_ * capsule_dim_ + i * feature_size * capsule_dim_);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleReshapeLayer);

}  // namespace caffe
