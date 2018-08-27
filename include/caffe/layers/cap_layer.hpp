#ifndef CAFFE_CAP_LAYER_HPP_
#define CAFFE_CAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class CapLayer : public Layer<Dtype> {
  public:
    explicit CapLayer(const LayerParameter& param):Layer<Dtype>(param){
      //phase_ = param.phase();
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "Cap"; }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    //Phase phase_;
    int input_capsule_dim_;
    int input_capsule_num_;
    int output_capsule_dim_;
    int output_capsule_num_;
    int M_; // batch size
    bool bias_term_;
    // store intermediate data that has the same dimension of bottom data
    Blob<Dtype> bottom_copy_;
    // store transposed u
    Blob<Dtype> u_T_;
    Blob<Dtype> bias_multiplier_;
    Blob<Dtype> u_;
    Blob<Dtype> b_;
    Blob<Dtype> c_;
    Blob<Dtype> s_;
    Blob<Dtype> v_;
    // sum, max_val are intermediate Blobs for calculating softmax values.
    Blob<Dtype> scale_;
};

}
#endif

