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
Dtype cap_len(const Dtype* s, const int len, Dtype* v){
  caffe_powx(len, s, Dtype(2), v);
  Dtype squared_norm = caffe_cpu_asum(len, v) + 1e-07;
  return squared_norm;
}

template <typename Dtype>
void SquashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* v_data = v_.mutable_cpu_data();
  Dtype len;
  for (int i = 0; i < M_; ++i) {
    for (int j = 0; j < capsule_num_; ++j) {
        len = cap_len(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
		     v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
	Dtype tmp = (1.0 - exp(-sqrt(len + 1e-7))) / sqrt(len + 1e-7);
        //if(i == 0 && capsule_num_ == 10)
          //LOG(INFO) << "cap len : "<<j<<", j: "<<len;
	caffe_cpu_scale(capsule_dim_, tmp, bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, top_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
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
    Dtype* v_data = v_.mutable_cpu_data();
    Dtype len;
    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < capsule_num_; ++j) {
        len = cap_len(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
		     v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);

	// new squash function from paper: Capsule Network Performance on Complex Data
	Dtype tmp = exp(-sqrt(len + 1e-7)) / (len + 1e-7) - 1.0 / sqrt(len + 1e-7) / (len + 1e-7) + exp(-sqrt(len + 1e-7)) / sqrt(len + 1e-7) / (len + 1e-7);
	Dtype tmp1 = 1.0 / sqrt(len + 1e-7) - 1.0 / sqrt(len + 1e-7) * exp(-sqrt(len + 1e-7));
	caffe_cpu_scale(capsule_dim_, tmp, v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
	caffe_add_scalar(capsule_dim_, tmp1, bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);	
	caffe_mul(capsule_dim_, top_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, 
		bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, 
		bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
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
