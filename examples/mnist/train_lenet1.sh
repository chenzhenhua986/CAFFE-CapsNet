#!/usr/bin/env sh
set -e

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -gpu 3 $@ 
./build/tools/caffe train --solver=examples/mnist/lenet_solver1.prototxt -gpu 3
#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@ 
