#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/full_connected_capsule_solver.prototxt -gpu 1 $@ 
