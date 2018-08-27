#ifndef CAFFE_MARGIN_LOSS_LAYER_HPP_
#define CAFFE_MARGIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class MarginLossLayer : public LossLayer<Dtype> {
 public:
  explicit MarginLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MarginLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  // Store caplule lengths
  Blob<Dtype> cap_lens_;
  // Store tmporary data, has the same dimension as the bottom
  Blob<Dtype> tmp_;
  Dtype m_upper_bound_; 
  Dtype m_lower_bound_;
  Dtype lambda_;
  int M_;
  int num_class_; 
  int dim_; 
};


}  // namespace caffe

#endif  // CAFFE_MARGIN_LOSS_LAYER_HPP_
