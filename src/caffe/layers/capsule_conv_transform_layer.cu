#include <vector>

#include "caffe/layers/capsule_conv_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // shape of bottom: M_ * input_capsule_num_ * (input_h * input_w * input_capsule_dim_)
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // shape of top: M_ * (ouput_capsule_num_ * output_h_ * output_w_) * (input_capsule_num_ * kh_ * kw_) * output_capsule_dim_
  Dtype* top_data = top[0]->mutable_gpu_data();
  // shape of weight: output_capsule_num_ * (kh_ * kw_ * input_capsule_num) * output_capsule_dim, eg.,32*288*(4*4)=32*288*16
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;

  for(int b = 0; b < M_; ++b) {
    for(int k = 0; k < output_capsule_num_; ++k) {
      for(int i = 0; i < output_h_; ++i) {
        for(int j = 0; j < output_w_; ++j) {
          // calculate corresponding indices of bottom data
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < input_capsule_num_; ++s) {
            for(int m = row_offset; m < row_offset + kw_; ++m) {
              for(int n = col_offset; n < col_offset + kh_; ++n) {
	        int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	        int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
					(k * output_h_ * output_w_ + i * output_w_ + j) * (kh_ * kw_ * input_capsule_num_) + 
						s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	        int w_index = (k * kh_ * kw_ * input_capsule_num_ + s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
		// transform 4 by 4 input matrix to a 4 by 4 output matrix
		//caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, output_capsule_dim0_, (Dtype)1., 
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, input_capsule_dim1_, (Dtype)1., 
			bottom_data + index, weight + w_index, (Dtype)0., top_data + top_index);
	      }
	    }
	  }
	}
      }
    }
  }

  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, total_output_num * (kh_ * kw_ * input_capsule_num_) * total_output_dim, 1, (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}


template <typename Dtype>
void CapsuleConvTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int total_output_dim = output_capsule_dim0_ * output_capsule_dim1_;
  const int total_input_dim = input_capsule_dim0_ * input_capsule_dim1_;
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

    // Gradient with respect to weight
    for(int b = 0; b < M_; ++b) {
      for(int k = 0; k < output_capsule_num_; ++k) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;  
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = row_offset; m < row_offset + kw_; ++m) {
                for(int n = col_offset; n < col_offset + kh_; ++n) {
	          int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	          int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
					(k * output_h_ * output_w_ + i * output_w_ + j) * (kh_ * kw_ * input_capsule_num_) + 
						s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	          int w_index = (k * kh_ * kw_ * input_capsule_num_ + s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
		  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, input_capsule_dim0_, output_capsule_dim1_, output_capsule_dim0_, (Dtype)1., 
		  caffe_gpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim0_, (Dtype)1., 
			bottom_data + index, top_diff + top_index, (Dtype)1., weight_diff + w_index); 
		  } 
	        }
	      }
	    }
	  }
	}
      }
    }	

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, output_h_ * output_w_ * output_capsule_num_ * (kh_ * kw_ * input_capsule_num_) * total_output_dim, M_, (Dtype)1., bias_multiplier_.gpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    for(int b = 0; b < M_; ++b) {
      for(int k = 0; k < output_capsule_num_; ++k) {
        for(int i = 0; i < output_h_; ++i) {
          for(int j = 0; j < output_w_; ++j) {
            int row_offset = i * stride_;  
            int col_offset = j * stride_;  
            for(int s = 0; s < input_capsule_num_; ++s) {
              for(int m = row_offset; m < row_offset + kw_; ++m) {
                for(int n = col_offset; n < col_offset + kh_; ++n) {
	          int index = (b * total_input_num + s * input_h_ * input_w_ + m * input_w_ + n) * total_input_dim;
	          int top_index = (b * total_output_num * (kh_ * kw_ * input_capsule_num_) + 
					(k * output_h_ * output_w_ + i * output_w_ + j) * (kh_ * kw_ * input_capsule_num_) + 
						s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * total_output_dim;
	          int w_index = (k * kh_ * kw_ * input_capsule_num_ + s * kh_ * kw_ + (m - row_offset) * kw_ + (n - col_offset)) * (input_capsule_dim1_ * output_capsule_dim1_);
		  caffe_gpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, input_capsule_dim1_, output_capsule_dim1_, (Dtype)1., 
			top_diff + top_index, this->blobs_[0]->gpu_data() + w_index, (Dtype)1., bottom[0]->mutable_gpu_diff() + index); 
		  //caffe_gpu_gemm(CblasNoTrans, CblasTrans, output_capsule_dim0_, output_capsule_dim0_, output_capsule_dim1_, (Dtype)1., 
			//this->blobs_[0]->gpu_data() + w_index, top_diff + top_index, (Dtype)1., bottom[0]->mutable_gpu_diff() + index); 
			//this->blobs_[0]->gpu_data() + w_index, top_diff + top_index, (Dtype)0., bottom[0]->mutable_gpu_diff() + index); 
		  } 
	        }
	      }
	    }
	  }
	}
      }
    }	
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleConvTransformLayer);

} 
