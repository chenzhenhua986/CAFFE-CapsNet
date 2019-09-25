#ifndef CAFFE_TENSOR_LAYER_HPP_
#define CAFFE_TENSOR_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TensorLayer : public Layer<Dtype> {
 public:
  explicit TensorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Tensor"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int stride_;
  int kh_;
  int kw_;
  int group_;
  int input_h_;
  int input_w_;
  int output_h_;
  int output_w_;
  int input_capsule_dim_size_;
  int output_capsule_dim_size_;
  int input_capsule_dim0_;
  int input_capsule_dim1_;
  int input_capsule_dim2_;
  int output_capsule_dim0_;
  int output_capsule_dim1_;
  int output_capsule_dim2_;
  int input_capsule_num_;
  int output_capsule_num_;
  int M_; // batch size
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
};

}
#endif  
