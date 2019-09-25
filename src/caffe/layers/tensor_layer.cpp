#include <vector>
#include <cfloat>

#include "caffe/layers/tensor_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {

template <typename Dtype>
void TensorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  stride_ = this->layer_param_.capsule_conv_param().stride();
  kh_ = this->layer_param_.capsule_conv_param().kh();
  kw_ = this->layer_param_.capsule_conv_param().kw();
  input_capsule_num_ = this->layer_param_.capsule_conv_param().input_capsule_num();
  output_capsule_num_ = this->layer_param_.capsule_conv_param().output_capsule_num();
  input_h_ = this->layer_param_.capsule_conv_param().input_h();
  input_w_ = this->layer_param_.capsule_conv_param().input_w();
  output_h_ = (input_h_ - kh_) / stride_ + 1;
  output_w_ = (input_w_ - kw_) / stride_ + 1;
  bias_term_ = this->layer_param_.capsule_conv_param().bias_term();

  const BlobShape& input_capsule_shape = this->layer_param_.capsule_conv_param().input_capsule_shape();
  input_capsule_dim_size_ = input_capsule_shape.dim(0) * input_capsule_shape.dim(1) * input_capsule_shape.dim(2);
  const BlobShape& output_capsule_shape = this->layer_param_.capsule_conv_param().output_capsule_shape();
  output_capsule_dim_size_ = output_capsule_shape.dim(0) * output_capsule_shape.dim(1) * output_capsule_shape.dim(2);
  CHECK_EQ(input_capsule_shape.dim(0), output_capsule_shape.dim(0)) <<"Capsule matrix mismatch"; 
  CHECK_EQ(input_capsule_shape.dim(1), output_capsule_shape.dim(1)) <<"Capsule matrix mismatch"; 
  input_capsule_dim0_ = input_capsule_shape.dim(0);
  input_capsule_dim1_ = input_capsule_shape.dim(1);
  input_capsule_dim2_ = input_capsule_shape.dim(2);
  output_capsule_dim0_ = output_capsule_shape.dim(0);
  output_capsule_dim1_ = output_capsule_shape.dim(1);
  output_capsule_dim2_ = output_capsule_shape.dim(2);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(6);
    weight_shape[0] = input_capsule_num_;
    weight_shape[1] = output_capsule_num_;
    weight_shape[2] = kh_;
    weight_shape[3] = kw_;
    weight_shape[4] = input_capsule_dim0_; // or output_capsule_dim0_
    weight_shape[5] = input_capsule_dim2_ * output_capsule_dim2_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_conv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, output_capsule_num_ * output_h_* output_w_ * output_capsule_dim0_ * output_capsule_dim1_ * output_capsule_dim2_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.capsule_conv_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void TensorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  vector<int> top_capsule_shape(7);
  top_capsule_shape[0] = M_;
  top_capsule_shape[1] = output_capsule_num_;
  top_capsule_shape[2] = output_h_;
  top_capsule_shape[3] = output_w_;
  top_capsule_shape[4] = output_capsule_dim0_;
  top_capsule_shape[5] = output_capsule_dim1_;
  top_capsule_shape[6] = output_capsule_dim2_;
  top[0]->Reshape(top_capsule_shape);

}


template <typename Dtype>
void TensorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(M_ * total_output_num * output_capsule_dim_size_, (Dtype)0., top_data);
  //LOG(INFO) << " forward output num: "<<output_capsule_num_;
  for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < output_capsule_num_; ++p) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;
  
	  //int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_;
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = 0; m < kh_; ++m) {
              for(int n = 0; n < kw_; ++n) {
                for(int c = 0; c < input_capsule_dim0_; ++c) {
		  int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim0_ * input_capsule_dim2_ * output_capsule_dim2_ + 
			c * input_capsule_dim2_ * output_capsule_dim2_;
		  int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_ + 
			c * input_capsule_dim1_ * input_capsule_dim2_; 
	          int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_ + 
			c * output_capsule_dim1_ * output_capsule_dim2_;
                 
                  // (2 * (4 * 2) * (2 * (2 * 4)) = 2 * (4 * 4)
		  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim2_, input_capsule_dim2_, (Dtype)1., 
			bottom_data + idx, weight + w_idx, (Dtype)1., top_data + top_idx);
                }
	      }
            }
          }
        }
      }
    }
  }

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, total_output_num * output_capsule_dim_size_, 1, (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  } 
}

template <typename Dtype>
void TensorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    for(int b = 0; b < M_; ++b) {
      for(int p = 0; p < output_capsule_num_; ++p) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;
    
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = 0; m < kh_; ++m) {
                for(int n = 0; n < kw_; ++n) {
                  for(int c = 0; c < input_capsule_dim0_; ++c) {
		    int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim0_ * input_capsule_dim2_ * output_capsule_dim2_ + 
		          c * input_capsule_dim2_ * output_capsule_dim2_;
		    int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_ + 
			  c * input_capsule_dim1_ * input_capsule_dim2_; 
	            int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_ + 
			  c * output_capsule_dim1_ * output_capsule_dim2_;
                 
                    // (2 * (4 * 2)^T * (2 * (4 * 4)) = (2 * (2 * 4))
		    caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim2_, output_capsule_dim2_, input_capsule_dim1_, (Dtype)1.0, 
			  bottom_data + idx, top_diff + top_idx, (Dtype)1., this->blobs_[0]->mutable_cpu_diff() + w_idx);
                  }
	        }
              }
            }
          }
        }
      }
    }

  }	
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, output_h_ * output_w_ * output_capsule_num_ * output_capsule_dim_size_, M_, (Dtype)1., bias_multiplier_.cpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    caffe_set(M_ * total_input_num * input_capsule_dim_size_, (Dtype)0., bottom_diff);
    // Gradient with respect to bottom data
    for(int b = 0; b < M_; ++b) {
      for(int p = 0; p < output_capsule_num_; ++p) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;
    
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = 0; m < kh_; ++m) {
                for(int n = 0; n < kw_; ++n) {
                  for(int c = 0; c < input_capsule_dim0_; ++c) {
		    int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim0_ * input_capsule_dim2_ * output_capsule_dim2_ + 
		          c * input_capsule_dim2_ * output_capsule_dim2_;
		    int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_ + 
			  c * input_capsule_dim1_ * input_capsule_dim2_; 
	            int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_ + 
			  c * output_capsule_dim1_ * output_capsule_dim2_;
                 
                    // (2 * (4 * 4)) * (2 * (2 * 4)^T) = (2 * (4 * 2))
		    caffe_cpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim1_, input_capsule_dim2_, output_capsule_dim2_, (Dtype)1., 
			  top_diff + top_idx, this->blobs_[0]->cpu_data() + w_idx, (Dtype)1., bottom_diff + idx);
                  }
	        }
              }
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TensorLayer);
#endif

INSTANTIATE_CLASS(TensorLayer);
REGISTER_LAYER_CLASS(Tensor);
}
