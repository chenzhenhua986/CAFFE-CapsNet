#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/baseline_solver.prototxt -gpu 1 $@ 
