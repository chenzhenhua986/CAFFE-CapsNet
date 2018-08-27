#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/g_capsule_solver.prototxt -gpu 0 $@ 
