#include <algorithm>
#include <vector>

#include "caffe/layers/squash_layer.hpp"

namespace caffe {

template <typename Dtype>
void SquashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  capsule_dim_ = this->layer_param_.squash_param().capsule_dim();
  const int total_dim = bottom[0]->count(1);
  capsule_num_ = total_dim / capsule_dim_;
}

template <typename Dtype>
void SquashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  v_.Reshape(bottom_shape);
  M_ = bottom[0]->count(0, 1);
  top[0]->Reshape(bottom_shape);
}

template<typename Dtype>
void squash(const Dtype* s, const int len, Dtype* v, Dtype* sum){
  caffe_copy(len, s, v);
  caffe_powx(len, v, Dtype(2), sum);
  Dtype squared_norm = caffe_cpu_asum(len, sum) + 1e-07;
  //LOG(INFO) << "norm: "<<squared_norm;
  Dtype coefficient = (squared_norm / (squared_norm + 1)) / sqrt(squared_norm);
  //LOG(INFO) << "coefficient: "<<coefficient;
  caffe_scal(len, coefficient, v);
}


template<typename Dtype>
Dtype cap_len(const Dtype* s, const int len, Dtype* v){
  caffe_powx(len, s, Dtype(2), v);
  Dtype squared_norm = caffe_cpu_asum(len, v) + 1e-07;
  return sqrt(squared_norm);
}

template <typename Dtype>
void SquashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* v_data = v_.mutable_cpu_data();
  for (int i = 0; i < M_; ++i) {
    for (int j = 0; j < capsule_num_; ++j) {
      squash(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
	top_data + i * capsule_num_*capsule_dim_ + j * capsule_dim_, v_data + 
		i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
    }
  }
}

template <typename Dtype>
void SquashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype len;
    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < capsule_num_; ++j) {
        len = cap_len(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
		bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
	Dtype tmp = len/(1.0+pow(len, 2.0));
        for (int k = 0; k < capsule_dim_; ++k) {
          bottom_diff[i * capsule_num_ * capsule_dim_ + j * capsule_dim_ + k] = (1.0 / tmp - 2.0 * len) / pow((1.0+pow(len, 2.0)), 2.0) * top_diff[i * capsule_num_ * capsule_dim_ + j * capsule_dim_ + k] + tmp;
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SquashLayer);
#endif

INSTANTIATE_CLASS(SquashLayer);
REGISTER_LAYER_CLASS(Squash);

}  // namespace caffe
