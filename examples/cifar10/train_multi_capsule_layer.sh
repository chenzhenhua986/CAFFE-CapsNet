#!/usr/bin/env sh
set -e
./build/tools/caffe train --solver=examples/cifar10/multi_capsule_layer_solver.prototxt -gpu 1$@ 
