#include <algorithm>
#include <vector>

#include "caffe/layers/capsule_reshape_layer.hpp"

namespace caffe {
template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  capsule_dim_ = this->layer_param_.capsule_reshape_param().capsule_dim();
}

template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  capsule_num_ = bottom[0]->count(1) / capsule_dim_;
  // for example, last layer's output is 256*6*6, then we need aplit it into32*8*36 and then transpose each 8*32 matrix
  group_num_ = bottom[0]->count(1, 2) / capsule_dim_;
  vector<int> top_shape(3);
  top_shape[0] = M_;
  top_shape[1] = capsule_num_;
  top_shape[2] = capsule_dim_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void CapsuleReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(CapsuleReshapeLayer);
#endif

INSTANTIATE_CLASS(CapsuleReshapeLayer);
REGISTER_LAYER_CLASS(CapsuleReshape);

}  // namespace caffe
