#include <vector>
#include <cfloat>

#include "device_functions.h"
#include "caffe/layers/capsule_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "gtest/gtest.h"
#include <stdlib.h>
#include <math.h>

namespace caffe {


  template <typename Dtype>
  __global__ void filterForward(const int n, const Dtype* bottom_data, 
    Dtype* top_data, const Dtype* weight, const int kh_, const int kw_, const int input_h_, const int input_w_,
    const int output_h_, const int output_w_, const int stride_, const int M_,
    const int input_capsule_num_, const int output_capsule_num_, const int output_capsule_dim_size_,
    const int input_capsule_dim1_, const int output_capsule_dim1_,
    const int input_capsule_dim_size_, const int input_capsule_dim0_,
    const int total_input_num, const int total_output_num)
  {
  
    CUDA_KERNEL_LOOP(index, n) {
      // printf("%u\n", index);
      // printf("--------");
  
      const int s = index % input_capsule_num_;
      const int s_base = index / input_capsule_num_;
  
      const int j = s_base % output_w_;
      const int j_base = index / (output_w_ * input_capsule_num_);
  
      const int i = j_base % output_h_;
      const int i_base = index / (output_h_ * output_w_ * input_capsule_num_);
  
      const int p = i_base % output_capsule_num_;
      const int p_base = index / (output_capsule_num_ * output_h_ * output_w_ * input_capsule_num_);
  
      const int b = p_base % M_;
  
      const int row_offset = i * stride_;  
      const int col_offset = j * stride_;
    
      // const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
      // const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
      int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_;
    //   // // // (1 * (2 * 4) * (4 * 8) = 2 * 8

      for(int m = 0; m < kh_; ++m) {
        for(int n = 0; n < kw_; ++n) {
          int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
          int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_;
          
          // matrix multiplication
          // mat1: input_capsule_dim0_ x input_capsule_dim1_
          // mat2: input_capsule_dim1_ x output_capsule_dim1_
          const Dtype* mat1 = bottom_data + idx;
          const Dtype* mat2 = weight + w_idx;
          Dtype* output = top_data + top_idx;
  
          for (int od1 = 0; od1 < output_capsule_dim1_; ++od1) {
            for (int id0 = 0; id0 < input_capsule_dim0_; ++id0) {
              Dtype current_r=  0;
              for (int id1 = 0; id1 < input_capsule_dim1_; ++id1) {       
                current_r+=mat1[id0*input_capsule_dim1_+id1]*mat2[id1*output_capsule_dim1_+od1];
              }
              // printf("r %f\n", current_r);
              // output[id0*output_capsule_dim1_+od1]+=current_r;
              atomicAdd((float*)(output+id0*output_capsule_dim1_+od1), (float)current_r);
              // printf("op %f\n", output[id0*output_capsule_dim1_+od1]);
            }
          }
  
        }
      }
    
    }
  
  }

  template <typename Dtype>
__global__ void filterBackwardCompute(const int n, const Dtype* bottom_data, 
  Dtype* weight, const Dtype* top_diff, const int kh_, const int kw_, const int input_h_, const int input_w_,
  const int output_h_, const int output_w_, const int stride_, const int M_,
  const int input_capsule_num_, const int output_capsule_num_, const int output_capsule_dim_size_,
  const int input_capsule_dim1_, const int output_capsule_dim1_,
  const int input_capsule_dim_size_, const int input_capsule_dim0_,
  const int total_input_num, const int total_output_num)
{
  CUDA_KERNEL_LOOP(index, n) {

    const int s = index % input_capsule_num_;
    const int s_base = index / input_capsule_num_;

    const int j = s_base % output_w_;
    const int j_base = index / (output_w_ * input_capsule_num_);

    const int i = j_base % output_h_;
    const int i_base = index / (output_h_ * output_w_ * input_capsule_num_);

    const int p = i_base % output_capsule_num_;
    const int p_base = index / (output_capsule_num_ * output_h_ * output_w_ * input_capsule_num_);

    const int b = p_base % M_;

    const int row_offset = i * stride_;  
    const int col_offset = j * stride_;

    // const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
    // const int total_output_num = output_capsule_num_ * output_h_ * output_w_;

    int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_;

    for(int m = 0; m < kh_; ++m) {
      for(int n = 0; n < kw_; ++n) {
        int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
        int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_;
        
        // matrix multiplication
        // mat1: input_capsule_dim0_ x input_capsule_dim1_
        // mat2: input_capsule_dim0_ x output_capsule_dim1_
        const Dtype* mat1 = bottom_data + idx;
        const Dtype* mat2 = top_diff + top_idx;
        Dtype* output = weight + w_idx;

        for (int od1 = 0; od1 < output_capsule_dim1_; ++od1) {
          for (int id1 = 0; id1 < input_capsule_dim1_; ++id1) {
            Dtype current_r= 0;
            for (int id0 = 0; id0 < input_capsule_dim0_; ++id0) {       
              current_r+=mat1[id0*input_capsule_dim1_+id1]*mat2[id0*output_capsule_dim1_+od1];
            }
            // output[id1*output_capsule_dim1_+od1]+=current_r;
            atomicAdd((float*)(output+id1*output_capsule_dim1_+od1), (float)current_r);
          }
        }

        
      }
    }

    
  }
}

template <typename Dtype>
__global__ void filterBackwardPass(const int n, const Dtype* top_diff, 
  Dtype* bottom_diff, const Dtype* weight, const int kh_, const int kw_, const int input_h_, const int input_w_,
  const int output_h_, const int output_w_, const int stride_, const int M_,
  const int input_capsule_num_, const int output_capsule_num_, const int output_capsule_dim_size_,
  const int input_capsule_dim1_, const int output_capsule_dim1_,
  const int input_capsule_dim_size_, const int output_capsule_dim0_,
  const int total_input_num, const int total_output_num)
{
  CUDA_KERNEL_LOOP(index, n) {

    const int s = index % input_capsule_num_;
    const int s_base = index / input_capsule_num_;

    const int j = s_base % output_w_;
    const int j_base = index / (output_w_ * input_capsule_num_);

    const int i = j_base % output_h_;
    const int i_base = index / (output_h_ * output_w_ * input_capsule_num_);

    const int p = i_base % output_capsule_num_;
    const int p_base = index / (output_capsule_num_ * output_h_ * output_w_ * input_capsule_num_);

    const int b = p_base % M_;

    const int row_offset = i * stride_;  
    const int col_offset = j * stride_;

    // const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
    // const int total_output_num = output_capsule_num_ * output_h_ * output_w_;

    int top_idx = (b * total_output_num + p * output_h_ * output_w_ + i * output_w_ + j) * output_capsule_dim_size_;

    for(int m = 0; m < kh_; ++m) {
      for(int n = 0; n < kw_; ++n) {
        int w_idx = (p * input_capsule_num_ * kh_ * kw_ + s * kw_ * kh_ + m * kw_ + n) * input_capsule_dim1_ * output_capsule_dim1_;
        int idx = (b * total_input_num + s * input_h_ * input_w_ + (row_offset + m) * input_w_ + col_offset + n) * input_capsule_dim_size_;
        // matrix multiplication
        // mat1: input_capsule_dim0_ x input_capsule_dim1_
        // mat2: input_capsule_dim1_ x output_capsule_dim1_
        const Dtype* mat1 = top_diff + top_idx;
        const Dtype* mat2 = weight + w_idx;
        Dtype* output = bottom_diff + idx;
        
        for (int id1 = 0; id1 < input_capsule_dim1_; ++id1) {
          for (int id0 = 0; id0 < output_capsule_dim0_; ++id0) {
            Dtype current_r= 0;
            for (int od1 = 0; od1 < output_capsule_dim1_; ++od1) {       
              current_r+=mat1[id0*output_capsule_dim1_+od1]*mat2[id1*output_capsule_dim1_+od1];
            }
            // output[id0*input_capsule_dim1_+id1]+=current_r;
            atomicAdd((float*)(output+id0*input_capsule_dim1_+id1), (float)current_r);
          }
        }

      }
    }
  }
}






template <typename Dtype>
void CapsuleConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(M_ * total_output_num * output_capsule_dim_size_, (Dtype)0., top_data);

  const int num_kernels=M_*output_capsule_num_*output_h_*output_w_*input_capsule_num_;
  filterForward<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
  (num_kernels, bottom_data, top_data, weight, kh_, kw_, input_h_, input_w_, output_h_, 
   output_w_, stride_, M_, input_capsule_num_, output_capsule_num_, output_capsule_dim_size_, 
   input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim_size_,
   input_capsule_dim0_, total_input_num, total_output_num);
  CUDA_POST_KERNEL_CHECK;


  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, total_output_num * output_capsule_dim_size_, 1, (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  } 

  
}

template <typename Dtype>
void CapsuleConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int total_input_num = input_capsule_num_ * input_h_ * input_w_;
  const int total_output_num = output_capsule_num_ * output_h_ * output_w_;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data(); 
   
    Dtype* weight = this->blobs_[0]->mutable_gpu_diff();
    const int num_kernels=M_*output_capsule_num_*output_h_*output_w_*input_capsule_num_;
    filterBackwardCompute<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    (num_kernels, bottom_data, weight, top_diff, kh_, kw_, input_h_, input_w_, output_h_, 
      output_w_, stride_, M_, input_capsule_num_, output_capsule_num_, output_capsule_dim_size_, 
      input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim_size_,
    input_capsule_dim0_, total_input_num, total_output_num);
    CUDA_POST_KERNEL_CHECK;

  }	
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, output_h_ * output_w_ * output_capsule_num_ * output_capsule_dim_size_, M_, (Dtype)1., bias_multiplier_.gpu_data(), top_diff, (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    caffe_gpu_set(M_ * total_input_num * input_capsule_dim_size_, (Dtype)0., bottom_diff);
    
    // Gradient with respect to bottom data
      
      const Dtype* weight = this->blobs_[0]->gpu_data();
      caffe_gpu_set(M_ * total_input_num * input_capsule_dim_size_, (Dtype)0., bottom_diff);
      const int num_kernels=M_*output_capsule_num_*output_h_*output_w_*input_capsule_num_;
      filterBackwardPass<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
      (num_kernels, top_diff, bottom[0]->mutable_gpu_diff(), this->blobs_[0]->gpu_data(), kh_, kw_, input_h_, input_w_, output_h_, 
        output_w_, stride_, M_, input_capsule_num_, output_capsule_num_, output_capsule_dim_size_, 
        input_capsule_dim1_, output_capsule_dim1_, input_capsule_dim_size_,
        output_capsule_dim0_, total_input_num, total_output_num);
      CUDA_POST_KERNEL_CHECK;

    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CapsuleConvLayer);

}

