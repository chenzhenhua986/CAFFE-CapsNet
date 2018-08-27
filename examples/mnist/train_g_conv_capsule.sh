#!/usr/bin/env sh
set -e
./build/tools/caffe train --solver=examples/mnist/g_conv_capsule_solver.prototxt -gpu 1 $@ 
