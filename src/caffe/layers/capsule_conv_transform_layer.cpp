#include <vector>
#include <cfloat>

#include "caffe/layers/capsule_conv_transform_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {

template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  stride_ = this->layer_param_.capsule_conv_transform_param().stride();
  kh_ = this->layer_param_.capsule_conv_transform_param().kh();
  kw_ = this->layer_param_.capsule_conv_transform_param().kw();
  input_capsule_num_ = this->layer_param_.capsule_conv_transform_param().input_capsule_num();
  output_capsule_num_ = this->layer_param_.capsule_conv_transform_param().output_capsule_num();
  input_h_ = this->layer_param_.capsule_conv_transform_param().input_h();
  input_w_ = this->layer_param_.capsule_conv_transform_param().input_w();
  output_h_ = (input_h_ - kh_) / stride_ + 1;
  output_w_ = (input_w_ - kw_) / stride_ + 1;
  bias_term_ = this->layer_param_.capsule_conv_transform_param().bias_term();

  const BlobShape& input_capsule_shape = this->layer_param_.capsule_conv_transform_param().input_capsule_shape();
  const BlobShape& output_capsule_shape = this->layer_param_.capsule_conv_transform_param().output_capsule_shape();
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
    weight_shape[0] = kh_ * kw_ * input_capsule_num_ * output_capsule_num_;
    weight_shape[1] = input_capsule_shape.dim(1);
    weight_shape[2] = output_capsule_shape.dim(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_conv_transform_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, output_capsule_num_ * output_h_* output_w_ * (kh_ * kw_ * input_capsule_num_) * output_capsule_dim0_ * output_capsule_dim1_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.capsule_conv_transform_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  vector<int> top_capsule_shape(4);
  top_capsule_shape[0] = M_;
  /*
  top_capsule_shape[1] = kh_ * kw_ * input_capsule_num_;
  top_capsule_shape[2] = output_h_ * output_w_ * output_capsule_num_;
  top_capsule_shape[3] = output_capsule_dim0_ * output_capsule_dim1_;
  */
  top_capsule_shape[1] = output_h_ * output_w_;
  top_capsule_shape[2] = kh_ * kw_ * input_capsule_num_;
  top_capsule_shape[3] = output_capsule_num_ * output_capsule_dim0_ * output_capsule_dim1_;
  top[0]->Reshape(top_capsule_shape);

  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  //LOG(INFO) << "backward in capsule transform layer ends1: "<<total_output_dim;
  //LOG(INFO) << "backward in capsule transform layer ends2: "<<total_input_dim;
  //LOG(INFO) << "backward in capsule transform layer ends3: "<<total_input_num;
  //LOG(INFO) << "backward in capsule transform layer ends4: "<<total_output_num;
  //LOG(INFO) << "backward in capsule transform layer ends4: "<<kw_<<kh_<<stride_;

  for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = row_offset; m < row_offset + kh_; ++m) {
              for(int n = col_offset; n < col_offset + kw_; ++n) {
	        int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	        int top_index = (b * output_h_ * output_w_ * (input_capsule_num_ * kh_ * kw_) + 
				(i * output_w_ + j) * (input_capsule_num_ * kw_ * kh_) +
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * output_capsule_num_ * total_output_dim;
	        int w_index = (s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_ * output_capsule_num_);
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_ * output_capsule_num_, input_capsule_dim1_, (Dtype)1., 
			bottom_data + index, weight + w_index, (Dtype)0., top_data + top_index);
	      }
	    }
	  }
	}
      }
    }

  /*
  for(int b = 0; b < M_; ++b) {
    for(int k = 0; k < output_capsule_num_; ++k) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = row_offset; m < row_offset + kh_; ++m) {
              for(int n = col_offset; n < col_offset + kw_; ++n) {
	        int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	        int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
				 k * (output_h_ * output_w_) * (kh_ * kw_) * input_capsule_num_ +
				(i * output_w_ + j) * (kw_ * kh_) * input_capsule_num_ +
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	        int w_index = (k * (kh_ * kw_) * input_capsule_num_ + 
			       s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
                //LOG(INFO) << "b, k, i, j, s, m, n, row_offset, col_offset: "<<b<<", "<<k<<", "<<i<<", "<<j<<", "<<s<<", "<<m<<", "<<n<<", "<<row_offset<<", "<<col_offset;
                //LOG(INFO) << "bottom index: "<<index;
                //LOG(INFO) << "top index: "<<top_index;
                //LOG(INFO) << "weight index: "<<w_index;
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, input_capsule_dim1_, (Dtype)1., 
			bottom_data + index, weight + w_index, (Dtype)0., top_data + top_index);
	      }
	    }
	  }
	}
      }
    }
  }
  */

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, total_output_num * (kh_ * kw_ * input_capsule_num_) * total_output_dim, 1, (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  } 
}

template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = row_offset; m < row_offset + kh_; ++m) {
              for(int n = col_offset; n < col_offset + kw_; ++n) {
	        int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	        int top_index = (b * output_h_ * output_w_ * (input_capsule_num_ * kh_ * kw_) + 
				(i * output_w_ + j) * (input_capsule_num_ * kw_ * kh_) +
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * output_capsule_num_ * total_output_dim;
	        int w_index = (s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_ * output_capsule_num_);
		  caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim1_ * output_capsule_num_, input_capsule_dim0_, (Dtype)1., 
			bottom_data + index, top_diff + top_index, (Dtype)1., weight_diff + w_index); 
	      }
	    }
	  }
	}
      }
    }
/*
    // Gradient with respect to weight
    for(int b = 0; b < M_; ++b) {
      for(int k = 0; k < output_capsule_num_; ++k) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;  
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = row_offset; m < row_offset + kh_; ++m) {
                for(int n = col_offset; n < col_offset + kw_; ++n) {
	          int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	          int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
				   k * (output_h_ * output_w_) * (kh_ * kw_) * input_capsule_num_ + 
				  (i * output_w_ + j) * (kw_ * kh_) * input_capsule_num_ +
				   s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	          int w_index = (k * (kh_ * kw_) * input_capsule_num_ + 
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
		  caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim0_, (Dtype)1., 
			bottom_data + index, top_diff + top_index, (Dtype)1., weight_diff + w_index); 
		} 
	      }
	    }
	  }
	}
      }
    }
*/
  }	
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, output_h_ * output_w_ * output_capsule_num_ * (kh_ * kw_ * input_capsule_num_) * total_output_dim, M_, (Dtype)1., bias_multiplier_.cpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
  for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = row_offset; m < row_offset + kh_; ++m) {
              for(int n = col_offset; n < col_offset + kw_; ++n) {
	        int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	        int top_index = (b * output_h_ * output_w_ * (input_capsule_num_ * kh_ * kw_) + 
				(i * output_w_ + j) * (input_capsule_num_ * kw_ * kh_) +
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * output_capsule_num_ * total_output_dim;
	        int w_index = (s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_ * output_capsule_num_);
		  caffe_cpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, input_capsule_dim1_, output_capsule_dim1_ * output_capsule_num_, (Dtype)1., 
			top_diff + top_index, this->blobs_[0]->cpu_data() + w_index, (Dtype)0., bottom[0]->mutable_cpu_diff() + index); 
			//top_diff + top_index, this->blobs_[0]->cpu_data() + w_index, (Dtype)1., bottom[0]->mutable_cpu_diff() + index); 
	      }
	    }
	  }
	}
      }
    }
    }
  /*
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    for(int b = 0; b < M_; ++b) {
      for(int k = 0; k < output_capsule_num_; ++k) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;  
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = row_offset; m < row_offset + kh_; ++m) {
                for(int n = col_offset; n < col_offset + kw_; ++n) {
	          int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	          int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
				   k * (output_h_ * output_w_) * (kh_ * kw_) * input_capsule_num_ + 
				  (i * output_w_ + j) * (kw_ * kh_) * input_capsule_num_ +
				   s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	          int w_index = (k * (kh_ * kw_) * input_capsule_num_ + 
				 s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
	          //int w_index = (k * kh_ * kw_ * input_capsule_num_ + s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
		  caffe_cpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, input_capsule_dim1_, output_capsule_dim1_, (Dtype)1., 
			top_diff + top_index, this->blobs_[0]->cpu_data() + w_index, (Dtype)1., bottom[0]->mutable_cpu_diff() + index); 
		  } 
	        }
	      }
	    }
	  }
	}
      }
    }
   */	
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleConvTransformLayer);
#endif

INSTANTIATE_CLASS(CapsuleConvTransformLayer);
REGISTER_LAYER_CLASS(CapsuleConvTransform);

}
