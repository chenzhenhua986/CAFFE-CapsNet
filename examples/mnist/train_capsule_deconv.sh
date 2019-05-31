#!/usr/bin/env sh
set -e
./build/tools/caffe train --solver=examples/mnist/capsule_deconv_solver.prototxt -gpu 0 $@ 
