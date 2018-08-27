#ifndef CAFFE_CAPSULE_RESHAPE_LAYER_HPP_
#define CAFFE_CAPSULE_RESHAPE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class CapsuleReshapeLayer : public NeuronLayer<Dtype> {
 public:
    explicit CapsuleReshapeLayer(const LayerParameter& param):NeuronLayer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CapsuleReshape"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int capsule_dim_;
  int capsule_num_;
  int group_num_;
  int M_;
};

}  // namespace caffe

#endif  // CAFFE_CAPSULE_RESHAPE_LAYER_HPP_
