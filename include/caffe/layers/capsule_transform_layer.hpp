#ifndef CAFFE_CAPSULE_TRANSFORM_LAYER_HPP_
#define CAFFE_CAPSULE_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class CapsuleTransformLayer : public Layer<Dtype> {
  public:
    explicit CapsuleTransformLayer(const LayerParameter& param):Layer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "CapsuleTransform"; }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // input: 1152*160=input_capsule_num_*(output_capsule_dim_*output_capsule_num_)
    int input_capsule_dim_;
    int input_capsule_num_;
    int output_capsule_dim_;
    int output_capsule_num_;
    int M_; // batch size
    bool bias_term_;
    Blob<Dtype> bias_multiplier_;
};

}
#endif

