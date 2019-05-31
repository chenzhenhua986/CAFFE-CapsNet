#include <vector>
#include <cfloat>

#include "caffe/layers/capsule_deconv_transform_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {

template <typename Dtype>
void CapsuleDeconvTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  stride_ = this->layer_param_.capsule_deconv_transform_param().stride();
  kh_ = this->layer_param_.capsule_deconv_transform_param().kh();
  kw_ = this->layer_param_.capsule_deconv_transform_param().kw();
  input_capsule_num_ = this->layer_param_.capsule_deconv_transform_param().input_capsule_num();
  output_capsule_num_ = this->layer_param_.capsule_deconv_transform_param().output_capsule_num();
  input_h_ = this->layer_param_.capsule_deconv_transform_param().input_h();
  input_w_ = this->layer_param_.capsule_deconv_transform_param().input_w();
  output_h_ = (input_h_ - 1) * stride_ + kh_;
  output_w_ = (input_w_ - 1) * stride_ + kw_;
  bias_term_ = this->layer_param_.capsule_deconv_transform_param().bias_term();

  const BlobShape& input_capsule_shape = this->layer_param_.capsule_deconv_transform_param().input_capsule_shape();
  const BlobShape& output_capsule_shape = this->layer_param_.capsule_deconv_transform_param().output_capsule_shape();
  CHECK_EQ(input_capsule_shape.dim(0), output_capsule_shape.dim(0)) <<"Capsule matrix mismatch"; 
  input_capsule_dim0_ = input_capsule_shape.dim(0);
  input_capsule_dim1_ = input_capsule_shape.dim(1);
  output_capsule_dim0_ = output_capsule_shape.dim(0);
  output_capsule_dim1_ = output_capsule_shape.dim(1);
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
    vector<int> weight_shape(3);
    //weight_shape[0] = kh_ * kw_ * input_capsule_num_ * output_capsule_num_;
    //weight_shape[1] = input_capsule_shape.dim(1);
    //weight_shape[2] = output_capsule_shape.dim(1);
    weight_shape[0] = kh_ * kw_ * output_capsule_num_;
    weight_shape[1] = input_capsule_shape.dim(1);
    weight_shape[2] = output_capsule_shape.dim(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_deconv_transform_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, output_capsule_num_ * output_h_* output_w_ * output_capsule_dim0_ * output_capsule_dim1_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.capsule_deconv_transform_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleDeconvTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  vector<int> top_capsule_shape(4);
  //top_capsule_shape[0] = M_;
  //top_capsule_shape[1] = output_h_ * output_w_;
  //top_capsule_shape[2] = kh_ * kw_ * input_capsule_num_;
  //top_capsule_shape[3] = output_capsule_num_ * output_capsule_dim0_ * output_capsule_dim1_;
  top_capsule_shape[0] = M_;
  top_capsule_shape[1] = output_capsule_num_;
  top_capsule_shape[2] = output_h_ * output_w_;
  top_capsule_shape[3] = output_capsule_dim0_ * output_capsule_dim1_;
  //LOG(INFO) << "capsule deconv transform layer1: "<<top_capsule_shape[1];
  //LOG(INFO) << "capsule deconv transform layer1: "<<top_capsule_shape[2];
  //LOG(INFO) << "capsule deconv transform layer1: "<<top_capsule_shape[3];
  top[0]->Reshape(top_capsule_shape);

  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void CapsuleDeconvTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  //const int total_input_num = input_capsule_num_ * input_h_ * input_w_ * kh_ * kw_ * output_capsule_num_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  //LOG(INFO) << "backward in capsule transform layer ends4: "<<kw_<<kh_<<stride_;
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < output_capsule_num_; ++i) {
      for(int j = 0; j < input_h_; ++j) {
        for(int k = 0; k < input_w_; ++k) {
          int x = j * stride_;  
          int y = k * stride_;  
          for(int m = 0; m < kh_; ++m) {
            for(int n = 0; n < kw_; ++n) {
	      int idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_input_dim;
	      int w_idx = (i * kh_ * kw_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
	      int top_idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_output_dim;
	      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, input_capsule_dim1_, (Dtype)1., 
	          bottom_data + idx, weight + w_idx, (Dtype)1., top_data + top_idx);
	    }
	  }
	}
      }
    }
  }
  /*for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            for(int m = 0; m < kh_; ++m) {
              for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ * kh_ * kw_ * output_capsule_num_ + 
				i * input_w_ * kh_ * kw_ * output_capsule_num_ + j * kh_ * kw_ * output_capsule_num_ + 
					s * kh_ * kw_ + m * kw_ + n) * total_input_dim;
		int w_idx = (p * input_h_ * input_w_ * output_capsule_num_ + i * input_w_ * output_capsule_num_ + j * output_capsule_num_) * input_capsule_dim1_ * output_capsule_dim0_;
		int top_idx = (b * total_output_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + (col_offset + n)) * total_output_dim; 
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, input_capsule_dim1_, (Dtype)1., 
			bottom_data + idx, weight + w_idx, (Dtype)1., top_data + top_idx);
		}
	      }
	    }
          }
        }
      }
    }*/

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, total_output_num * total_output_dim, 1, (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  } 
}

template <typename Dtype>
void CapsuleDeconvTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  //const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  // Gradient with respect to weight
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < output_capsule_num_; ++i) {
      for(int j = 0; j < input_h_; ++j) {
        for(int k = 0; k < input_w_; ++k) {
          int x = j * stride_;  
          int y = k * stride_;  
          for(int m = 0; m < kh_; ++m) {
            for(int n = 0; n < kw_; ++n) {
	      int idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_input_dim;
	      int w_idx = (i * kh_ * kw_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
	      int top_idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_output_dim;
	      caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim0_, (Dtype)1., 
	  	  bottom_data + idx, top_diff + top_idx, (Dtype)1., weight_diff + w_idx);
	    }
	  }
	}
      }
    }
  }

  /*for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            for(int m = 0; m < kh_; ++m) {
              for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ * kh_ * kw_ * output_capsule_num_ + 
				i * input_w_ * kh_ * kw_ * output_capsule_num_ + j * kh_ * kw_ * output_capsule_num_ + 
					s * kh_ * kw_ + m * kw_ + n) * total_input_dim;
		int w_idx = (p * input_h_ * input_w_ * output_capsule_num_ + i * input_w_ * output_capsule_num_ + j * output_capsule_num_) * input_capsule_dim1_ * output_capsule_dim0_;
		int top_idx = (b * total_output_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + (col_offset + n)) * total_output_dim; 
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim0_, (Dtype)1., 
			bottom_data + idx, top_diff + top_idx, (Dtype)1., weight_diff + w_idx); 
		}
	      }
	    }
          }
        }
      }
    }*/


  }	
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, output_h_ * output_w_ * output_capsule_num_ * total_output_dim, M_, (Dtype)1., bias_multiplier_.cpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < output_capsule_num_; ++i) {
      for(int j = 0; j < input_h_; ++j) {
        for(int k = 0; k < input_w_; ++k) {
          int x = j * stride_;  
          int y = k * stride_;  
          for(int m = 0; m < kh_; ++m) {
            for(int n = 0; n < kw_; ++n) {
	      int idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_input_dim;
	      int w_idx = (i * kh_ * kw_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
	      int top_idx = (b * total_output_num + i * output_h_ * output_w_ + x * output_w_ + y) * total_output_dim;
	      caffe_cpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, input_capsule_dim1_, output_capsule_dim1_, (Dtype)1., 
	  	  top_diff + top_idx, this->blobs_[0]->cpu_data() + w_idx, (Dtype)0., bottom[0]->mutable_cpu_diff() + idx);
	    }
	  }
	}
      }
    }
  }
  /*for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            for(int m = 0; m < kh_; ++m) {
              for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ * kh_ * kw_ * output_capsule_num_ + 
				i * input_w_ * kh_ * kw_ * output_capsule_num_ + j * kh_ * kw_ * output_capsule_num_ + 
					s * kh_ * kw_ + m * kw_ + n) * total_input_dim;
		int w_idx = (p * input_h_ * input_w_ * output_capsule_num_ + i * input_w_ * output_capsule_num_ + j * output_capsule_num_) * input_capsule_dim1_ * output_capsule_dim0_;
		int top_idx = (b * total_output_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + (col_offset + n)) * total_output_dim; 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, input_capsule_dim1_, output_capsule_dim1_, (Dtype)1., 
			top_diff + top_idx, this->blobs_[0]->cpu_data() + w_idx, (Dtype)0., bottom[0]->mutable_cpu_diff() + idx); 
		}
	      }
	    }
          }
        }
      }
    }*/

  }
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleDeconvTransformLayer);
#endif

INSTANTIATE_CLASS(CapsuleDeconvTransformLayer);
REGISTER_LAYER_CLASS(CapsuleDeconvTransform);

}
