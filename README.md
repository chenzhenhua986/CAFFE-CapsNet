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

# CAFFE-CapsNet
This repository is for the paper "Generalized Capsule Networks with Trainable Routing Procedure". 

G-CapsNet makes the rouing prodedure become part of the training process, it supports both "full-connected" version of CapsNet as well as "convolutional" version of CapsNet. 

G-CapsNet is tested on MNIST and achieves comparible performance as normal CapsNet.

## Guidline on MNIST
To train a "full-connected" version of G-CapsNet, run
```
sh examples/mnist/train_full_connected_capsule.sh
```
To train a "convolutioal" version of G-CapsNet, run
```
sh examples/mnist/train_g_conv_capsule.sh 
```
To train a baseline (Please read the paper for more details), run
```
sh examples/mnist/train_baseline.sh 
```

## Guidline on CIFAR10
To train a "full-connected" version of G-capsNet, run
```
sh examples/cifar10/train_full_connected.sh
```
To train a baseline (Please read the paper for more details), run
```
sh examples/cifar10/train_baseline.sh
```

Please send email to chen478@iu.edu for questions.

## Citation

    @article{G-CapsNet,
      Author = {Chen, Zhenhua and Crandall, David},
      Journal = {https://arxiv.org/abs/1808.08692v1},
      Title = {Generalized Capsule Networks with Trainable Routing Procedure},
      Year = {2018}
    }


