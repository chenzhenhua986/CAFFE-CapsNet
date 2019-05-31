#include <vector>

#include "caffe/layers/capsule_deconv_routing_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>
#include <math.h>

namespace caffe {

/*
 * kernel = 3*3
 * in/out capsule_shape:4*4=16
 * conv intput to output shape 6*6 with 3*3  kernel output 2*2
 * deconv intput to output shape 2*2 with 3*3  kernel output 6*6
 *
 *conv example: input shape: 2*2*(128*3*3)*160
 *		w:128*3*3*1
 *		out:10*2*2*16
 *
 *deconv example: input: 10*2*2*16
 *		  w: 2*2 *(128*3*3)	
 *		  out: 2*2*(128*3*3)*160
 *
 *
 * */

template <typename Dtype>
void CapsuleDeconvRoutingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  stride_ = this->layer_param_.capsule_deconv_routing_param().stride();
  kh_ = this->layer_param_.capsule_deconv_routing_param().kh();
  kw_ = this->layer_param_.capsule_deconv_routing_param().kw();
  input_capsule_num_ = this->layer_param_.capsule_deconv_routing_param().input_capsule_num();
  output_capsule_num_ = this->layer_param_.capsule_deconv_routing_param().output_capsule_num();
  input_h_ = this->layer_param_.capsule_deconv_routing_param().input_h();
  input_w_ = this->layer_param_.capsule_deconv_routing_param().input_w();
  output_h_ = (input_h_ - 1) * stride_ + kh_;
  output_w_ = (input_w_ - 1) * stride_ + kw_;
  //const BlobShape& output_capsule_shape = this->layer_param_.capsule_deconv_routing_param().output_capsule_shape();
  //output_capsule_dim_size_ = output_capsule_shape.dim(0) * output_capsule_shape.dim(1);
  const BlobShape& input_capsule_shape = this->layer_param_.capsule_deconv_routing_param().input_capsule_shape();
  input_capsule_dim_size_ = input_capsule_shape.dim(0) * input_capsule_shape.dim(1);
  //LOG(INFO) << "out cap dim size: ";
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(2);
    //weight_shape[0] = input_h_ * input_w_;
    //weight_shape[1] = output_capsule_num_ * kh_ * kw_;
    weight_shape[0] = input_capsule_num_;
    weight_shape[1] = output_capsule_num_ * kh_ * kw_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.capsule_deconv_routing_param().weight_filler()));
    //LOG(INFO) << "gpu diff: "<<this->blobs_[0]->gpu_diff();
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CapsuleDeconvRoutingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0, 1);
  //vector<int> top_shape(3);
  //top_shape[0] = M_;
  //top_shape[1] = input_h_ * input_w_;
  //top_shape[2] = output_capsule_num_ * kh_ * kw_ * input_capsule_num_ * output_capsule_dim_size_;
  vector<int> top_shape(4);
  top_shape[0] = M_;
  top_shape[1] = output_capsule_num_;
  top_shape[2] = output_h_ * output_w_;
  top_shape[3] = input_capsule_dim_size_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CapsuleDeconvRoutingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            //for(int m = 0; m < kh_; ++m) {
              //for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ + i * input_w_ + j) * input_capsule_dim_size_;
		int w_idx = p * output_capsule_num_ * kh_ * kw_ + s * kw_ * kh_;
		int top_idx = (b * total_output_num + s * output_h_ * output_w_ + row_offset * output_w_ + col_offset) * input_capsule_dim_size_; 
		//int top_idx = (b * total_output_num + s * output_h_ * output_w_ + (row_offset + m) * output_w_ + (col_offset + n)) * input_capsule_dim_size_; 
                // (1 * 3 * 3)^T * (1 * 16) = 3*3 * 16
  		//LOG(INFO) << "out cap dim size top_idx: "<<top_idx;
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, kh_ * kw_, input_capsule_dim_size_, 1, (Dtype)1., 
			weight + w_idx, bottom_data + idx, (Dtype)1., top_data + top_idx);
		//caffe_cpu_gemm(CblasTrans, CblasNoTrans, input_capsule_dim_size_, kh_ * kw_, 1, (Dtype)1., 
		//	bottom_data + idx, weight + w_idx, (Dtype)1., top_data + top_idx);
	      //}
            //}
          }
        }
      }
    }
  }

  /*
  for(int b = 0; b < M_; ++b) {
    for(int i = 0; i < input_w_ * input_h_; ++i) { // input_w, input_h: 2, 2
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, output_capsule_dim_size_, (Dtype)1., input_capsule_num_ * input_h_ * input_w_, (Dtype)1., 
		bottom_data + b * (input_h_ * input_w_) * (input_capsule_num_ * output_capsule_dim_size_) + 
				i * (input_capsule_num_ * output_capsule_dim_size_), 
			weight + i * (kh_ * kw_ * output_capsule_num_), (Dtype)0., 
			top_data + b * input_h_ * input_w_ * (output_capsule_num_ * kh_ * kw_) * (input_capsule_num_ * output_capsule_dim_size_) + 
				i * (output_capsule_num_ * kh_ * kw_) * (input_capsule_num_ * output_capsule_dim_size_));
    }
  }*/
}

template <typename Dtype>
void CapsuleDeconvRoutingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight.
    /*for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < input_w_ * input_h_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (Dtype)1., (output_capsule_num_ * kh_ * kw_), input_capsule_num_ * output_capsule_dim_size_, (Dtype)1., 
		bottom_data + b * input_w_ * input_h_ * input_capsule_num_ * output_capsule_dim_size_ + i * input_capsule_num_ * output_capsule_dim_size_, 
			top_diff + b * input_w_ * input_h_ * (output_capsule_num_ * kh_ * kw_) * (input_capsule_num_ * output_capsule_dim_size_) + 
				   i * (output_capsule_num_ * kh_ * kw_) * input_capsule_num_ * output_capsule_dim_size_, (Dtype)1.0,  
					this->blobs_[0]->mutable_cpu_diff() + i * (kh_ * kw_ * output_capsule_num_));
      }
    }*/
  for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            //for(int m = 0; m < kh_; ++m) {
              //for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ + i * input_w_ + j) * input_capsule_dim_size_;
		int w_idx = p * output_capsule_num_ * kh_ * kw_ + s * kw_ * kh_;
		//int top_idx = (b * total_output_num + s * output_h_ * output_w_ + (row_offset + m) * output_w_ + (col_offset + n)) * input_capsule_dim_size_; 
		int top_idx = (b * total_output_num + s * output_h_ * output_w_ + row_offset * output_w_ + col_offset) * input_capsule_dim_size_; 
                // (1 * 3 * 3) * (3*3 * 16) = 1 * 16
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, input_capsule_dim_size_, kh_ * kw_, (Dtype)1.0, 
			bottom_data + idx, top_diff + top_idx, (Dtype)1., this->blobs_[0]->mutable_cpu_diff() + w_idx);
	      //}
            //}
          }
        }
      }
    }
  }

  }
//LOG(INFO) << "capsule deconv layer 3: "<<kh_;
//LOG(INFO) << "capsule deconv layer 3: "<<input_capsule_num_;
//LOG(INFO) << "capsule deconv layer 3: "<<output_capsule_num_;
//LOG(INFO) << "capsule deconv layer 3: "<<output_capsule_dim_size_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data.
    /*for(int b = 0; b < M_; ++b) {
      for(int i = 0; i < input_w_ * input_h_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  (kh_ * kw_ * input_capsule_num_), output_capsule_dim_size_, (Dtype)1.,  (Dtype)1., 
		this->blobs_[0]->cpu_data() + i * (kh_ * kw_ * output_capsule_num_),
			top_diff + b * input_w_ * input_h_ * (output_capsule_num_ * kh_ * kw_) * input_capsule_num_ * output_capsule_dim_size_ + 
				   i * (output_capsule_num_ * kh_ * kw_) * input_capsule_num_ * output_capsule_dim_size_, (Dtype)0., 
				bottom[0]->mutable_cpu_diff() + b * input_h_ * input_w_ * input_capsule_num_ * output_capsule_dim_size_ + 
					i * input_capsule_num_ * output_capsule_dim_size_);
      }
    }*/
  for(int b = 0; b < M_; ++b) {
    for(int p = 0; p < input_capsule_num_; ++p) {
      for(int i = 0; i < input_h_; ++i) {
        for(int j = 0; j < input_w_; ++j) {
          int row_offset = i * stride_;  
          int col_offset = j * stride_;  
          for(int s = 0; s < output_capsule_num_; ++s) {
            //for(int m = 0; m < kh_; ++m) {
              //for(int n = 0; n < kw_; ++n) {
	        int idx = (b * total_input_num + p * input_h_ * input_w_ + i * input_w_ + j) * input_capsule_dim_size_;
		int w_idx = p * output_capsule_num_ * kh_ * kw_ + s * kw_ * kh_;
		//int top_idx = (b * total_output_num + s * output_h_ * output_w_ + (row_offset + m) * output_w_ + (col_offset + n)) * input_capsule_dim_size_; 
		int top_idx = (b * total_output_num + s * output_h_ * output_w_ + row_offset * output_w_ + col_offset) * input_capsule_dim_size_; 
                // (1 * 16) * (3*3 * 16)^T = 1 * 3 * 3
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, kh_ * kw_, input_capsule_dim_size_, (Dtype)0., 
			this->blobs_[0]->cpu_data() + w_idx, top_diff + top_idx, (Dtype)1., bottom[0]->mutable_cpu_diff() + idx);
	      //}
            //}
          }
        }
      }
    }
  }

  }
}

#ifdef CPU_ONLY
STUB_GPU(CapsuleDeconvRoutingLayer);
#endif

INSTANTIATE_CLASS(CapsuleDeconvRoutingLayer);
REGISTER_LAYER_CLASS(CapsuleDeconvRouting);

}
