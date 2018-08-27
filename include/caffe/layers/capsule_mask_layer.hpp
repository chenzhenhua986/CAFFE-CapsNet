#ifndef CAFFE_CAPSULE_MASK_LAYER_HPP_
#define CAFFE_CAPSULE_MASK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Mask layer in the paper: Dynamic Routing between Capsules
 */
template <typename Dtype>
class CapsuleMaskLayer : public Layer<Dtype> {
 public:
  explicit CapsuleMaskLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CapsuleMask"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> v_;
  Blob<Dtype> max_;
  int class_num_;
  int capsule_dim_;
  int M_; // batch size
};

}  // namespace caffe

#endif  // CAFFE_CAPSULE_MASK_LAYER_HPP_
