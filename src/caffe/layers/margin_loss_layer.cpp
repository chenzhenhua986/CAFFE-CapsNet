#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layers/margin_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MarginLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  num_class_ = this->layer_param_.margin_param().num_class();
  m_upper_bound_ = this->layer_param_.margin_param().m_upper_bound();
  m_lower_bound_ = this->layer_param_.margin_param().m_lower_bound();
  lambda_ = this->layer_param_.margin_param().lambda();
}

template <typename Dtype>
void MarginLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  M_ = bottom[0]->count(0, 1);
  dim_ = bottom[0]->count(1) / num_class_;
  //vector<int> top_shape;
  vector<int> cap_lens_shape;
  //top_shape.push_back(M_);
  cap_lens_shape.push_back(M_);
  cap_lens_shape.push_back(num_class_);
  cap_lens_.Reshape(cap_lens_shape);
  tmp_.Reshape(bottom[0]->shape());
  //top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MarginLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  
  // Calculate norm for each capsule
  Dtype* lens = cap_lens_.mutable_cpu_data();
  Dtype* tmp = tmp_.mutable_cpu_data();
  Dtype loss = Dtype(0);
  //caffe_sqr(M_ * num_class_ * dim_, bottom_data, tmp);
  caffe_powx(M_ * num_class_ * dim_, bottom_data, Dtype(2), tmp);
  int l, T;
  for(int i = 0; i < M_; ++i) {
    l = static_cast<int>(label[i]);
    for(int j = 0; j < num_class_; ++j) {
      //caffe_sqr(dim_, bottom_data + i * num_class_ * dim_ + j * dim_, data);
      lens[i * num_class_ + j] = std::sqrt(caffe_cpu_asum(dim_, tmp + i * num_class_ * dim_ + j * dim_) + 1e-07);
      /*for(int k = 0; k < dim_; ++k) {
        LOG(INFO) << "bottom data : , "<< i * num_class_ * dim_ + j * dim_ + k <<", "<< bottom_data[i * num_class_ * dim_ + j * dim_ + k];
      }*/
      //LOG(INFO) << "tmp offset : "<< i * num_class_ + j << ", " << i * num_class_ * dim_ + j * dim_;
      //LOG(INFO) << "asum : "<< i * num_class_ + j << ", " << caffe_cpu_asum(dim_, tmp + i * num_class_ * dim_ + j * dim_);
      //LOG(INFO) << "lens : "<< i * num_class_ + j << ", " << lens[i * num_class_ + j];
      T = (l == j ? 1 : 0);
      loss += T * pow(std::max(Dtype(0), m_upper_bound_ - lens[i * num_class_ + j]), 2) + 
	lambda_ * (1 - T) * pow(std::max(Dtype(0), lens[i * num_class_ + j] - m_lower_bound_), 2); 
    }
    //LOG(INFO) << "margin loss : "<<i << ", " << loss / M_;
    //LOG(INFO) << "label : "<<i << ", " <<l;
  }
  //LOG(INFO) << "margin loss : " << loss;
  top[0]->mutable_cpu_data()[0] = (Dtype)loss / (Dtype)M_;
}  


template <typename Dtype>
void MarginLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* lens = cap_lens_.cpu_data();
    int l, T;
    for (int i = 0; i < M_; ++i) {
      l = static_cast<int>(label[i]);
      for(int j = 0; j < num_class_; ++j) {
	Dtype grad_scalar = Dtype(0);
        T = (l == j ? 1 : 0);
	
	if(T == 1 && lens[i * num_class_  + j] < m_upper_bound_) {
	  grad_scalar = (-2) * (m_upper_bound_ - lens[i * num_class_ + j]) / lens[i * num_class_ + j];
	  //grad_scalar = (-2) * (m_upper_bound_ - lens[i * num_class_ + j]);
	} 
	if(T == 0 && lens[i * num_class_  + j] > m_lower_bound_) {
	  grad_scalar = lambda_ * 2 * (lens[i * num_class_ + j] - m_lower_bound_) / lens[i * num_class_ + j];
	} 
	caffe_cpu_scale(dim_, grad_scalar, bottom_data + i * num_class_ * dim_ + j * dim_, bottom_diff + i * num_class_ * dim_ + j * dim_);
        //LOG(INFO) << "grad_scalar : "<<i * num_class_ + j<< ", " << grad_scalar;
        //LOG(INFO) << "label : "<<l <<", j: "<<j  << ", T" << T;
        //LOG(INFO) << "bottom : "<<j  << ", " << bottom_data[i * num_class_ + j];
        //LOG(INFO) << "lens : "<< i * num_class_ + j << ", " << lens[i * num_class_ + j];
        /*
	for(int k = 0; k < dim_; ++k) {
	  bottom_diff[i * num_class_ * dim_ + j * dim_ + k] = grad_scalar * bottom_data[i * num_class_ * dim_ + j * dim_ + k] / lens[i * num_class_  + j]; 
	}*/
      }
    }
  }
}

INSTANTIATE_CLASS(MarginLossLayer);
REGISTER_LAYER_CLASS(MarginLoss);

}  // namespace caffe
