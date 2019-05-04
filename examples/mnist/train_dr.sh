#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/dr_solver.prototxt -gpu 0 $@ 
