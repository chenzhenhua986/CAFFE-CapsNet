#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/squash_layer.hpp"

namespace caffe {
template<typename Dtype>
void cap_len(const Dtype* s, const int len, Dtype* v, Dtype* squared_norm){
  // caffe_gpu_memcpy(len, s, v);
  caffe_gpu_powx(len, s, Dtype(2), v);
  caffe_gpu_asum(len, v, squared_norm);
}

template <typename Dtype>
void SquashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* v_data = v_.mutable_gpu_data();
  Dtype len;
  for (int i = 0; i < M_; ++i) {
    for (int j = 0; j < capsule_num_; ++j) {
        cap_len(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
		     v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, &len);
	//Dtype tmp = len / ((Dtype)1. + len) / sqrt(len + 1e-7);
	// new squash function from paper: Capsule Network Performance on Complex Data
	Dtype tmp = (1.0 - exp(-sqrt(len + 1e-7))) / sqrt(len + 1e-7);
        //LOG(INFO) << "cap len : "<<j<<", j: "<<len;
	caffe_gpu_scale(capsule_dim_, tmp, bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, top_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
    }
  }
}

template <typename Dtype>
void SquashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* v_data = v_.mutable_gpu_data();
    //Dtype* v_diff = v_.mutable_gpu_diff();
    Dtype len;
    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < capsule_num_; ++j) {
        cap_len(bottom_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, capsule_dim_, 
		     v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, &len);

	// new squash function from paper: Capsule Network Performance on Complex Data
	Dtype tmp = exp(-sqrt(len + 1e-7)) / (len + 1e-7) - 1.0 / sqrt(len + 1e-7) / (len + 1e-7) + exp(-sqrt(len + 1e-7)) / sqrt(len + 1e-7) / (len + 1e-7);
	Dtype tmp1 = 1.0 / sqrt(len + 1e-7) - 1.0 / sqrt(len + 1e-7) * exp(-sqrt(len + 1e-7));
	//Dtype tmp = ((Dtype)1. - len) / pow(((Dtype)1. + len), 2.0) / sqrt(len + 1e-7);
	//Dtype tmp1 = len / ((Dtype)1. + len) / sqrt(len + 1e-7);
	caffe_gpu_scale(capsule_dim_, tmp, v_data + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);
	caffe_gpu_add_scalar(capsule_dim_, tmp1, bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);	
	caffe_gpu_mul(capsule_dim_, top_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, 
		bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_, 
		bottom_diff + i * capsule_num_ * capsule_dim_ + j * capsule_dim_);

        //LOG(INFO) << "cap len : "<<j<<", j: "<<len;
        //LOG(INFO) << "tmp : "<<j<<", j: "<<tmp;
        //LOG(INFO) << "tmp1 : "<<j<<", j: "<<tmp1;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SquashLayer);
}  // namespace caffe
