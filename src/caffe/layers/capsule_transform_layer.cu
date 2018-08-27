#include <vector>
#include <math.h> 
#include <cfloat>

#include "caffe/layers/capsule_transform_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapsuleTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int total_input = input_capsule_num_ * input_capsule_dim_;
  const int total_output = output_capsule_num_ * output_capsule_dim_;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // shape of weights: input_capsule_num * input_capsule_dim * (output_capsule_num * output_capsule_dim)
  const Dtype* weight = this->blobs_[0]->gpu_data();

  // Shape of input: input_capsule_num * input_capsule_dim, namely, 1152*8 
  // Shape of top: input_capsule_num * (output_capsule_num * output_capsule_dim), namely, 1152*(16*10)
  // Shape of weights: input_capsule_num * input_capsule_dim * (output_capsule_num * output_capsule_dim), namely, 1152*8*10*16
  // LOG(INFO) << "capsule transform forward phase1: "<<M_;
  for(int i = 0; i < M_; ++i) {
    for(int j = 0; j < input_capsule_num_; ++j) {
      // 1*8 multiply 8*(16*10) = 1*(16*10)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, total_output, input_capsule_dim_, (Dtype)1., 
	bottom_data + i * total_input + j * input_capsule_dim_, 
		weight + j * input_capsule_dim_ * total_output, (Dtype)0., 
			top_data + i * input_capsule_num_ * total_output + j * total_output);
    }
  }

  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, input_capsule_num_ *  total_output, 1, (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}


template <typename Dtype>
void CapsuleTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    // Gradient with respect to weight
    for(int i = 0; i < M_; ++i) {
      for(int j = 0; j < input_capsule_num_; ++j) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, input_capsule_dim_, output_capsule_num_ * output_capsule_dim_, 1,
            (Dtype)1., bottom_data + i * input_capsule_num_ * input_capsule_dim_ + j * input_capsule_dim_, 
			top_diff + i * input_capsule_num_ * output_capsule_num_ * output_capsule_dim_ + j * output_capsule_num_ * output_capsule_dim_, 
				(Dtype)1., weight_diff + j * input_capsule_dim_ * output_capsule_num_ * output_capsule_dim_);
				//(Dtype)1./ M_, weight_diff + j * input_capsule_dim_ * output_capsule_num_ * output_capsule_dim_);
      }
    }
  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    //caffe_gpu_gemv<Dtype>(CblasTrans, M_, input_capsule_num_ * output_capsule_num_ * output_capsule_dim_, (Dtype)1., top_diff, bias_multiplier_.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, input_capsule_num_ *  output_capsule_num_ * output_capsule_dim_, M_, (Dtype)1., bias_multiplier_.gpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    for(int i = 0; i < M_; ++i) {
      for(int j = 0; j < input_capsule_num_; ++j) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, input_capsule_dim_, output_capsule_num_ * output_capsule_dim_,
            (Dtype)1., top_diff + i * input_capsule_num_ * output_capsule_num_ * output_capsule_dim_ + j * output_capsule_num_ * output_capsule_dim_, 
		this->blobs_[0]->gpu_data() + j * input_capsule_dim_ * output_capsule_num_ * output_capsule_dim_,
          		(Dtype)0., bottom[0]->mutable_gpu_diff() + i * input_capsule_num_ * input_capsule_dim_ + j * input_capsule_dim_);
      }
    }
  }
  // LOG(INFO) << "backward in capsule transform layer ends: ";
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleTransformLayer);

} 
