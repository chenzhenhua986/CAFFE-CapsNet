# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }


# P-CapsNet
This part is for the paper [Capsule Networks without Routing procedures](https://openreview.net/forum?id=B1gNfkrYvS)  The primary code can been seen at [tensor_layer.cpp](https://github.com/chenzhenhua986/CAFFE-CapsNet/blob/master/src/caffe/layers/tensor_layer.cpp). For cuda-accelerated version, please see [capsule_conv_layer.cu](https://github.com/chenzhenhua986/CAFFE-CapsNet/blob/master/src/caffe/layers/capsule_conv_layer.cu). Note that the current acclerated version only supports the 2D tensor case. Below is an example of adding a capsule layer of P-CapsNets.

```
layer {
  name: "conv1"
  type: "Tensor"
  bottom: "data"
  top: "conv1"
  capsule_conv_param {
    weight_filler {
      type: "msra"
    }
    stride: 2
    kh: 3
    kw: 3
    input_capsule_num: 1
    output_capsule_num: 1
    output_capsule_shape {
      dim: 1
      dim: 1
      dim: 32
    }
    input_h: 28
    input_w: 28
    input_capsule_shape {
      dim: 1
      dim: 1
      dim: 1
    }
    bias_term: false
  }
}
```
To see a full example, please check [new_capsule_train.prototxt](https://github.com/chenzhenhua986/CAFFE-CapsNet/blob/master/examples/mnist/new_capsule_train.prototxt). To train a sample P-CapsNet model on MNIST, run
```
sh examples/mnist/train_new_capsule.sh
```



## Training CaspNets with dynamic routing on MNIST & CIFAR10
To train the CapsNet with dynamic routing in the orignal paper, run
```
sh examples/mnist/train_dr.sh 

sh examples/cifar10/train_baseline.sh
```






