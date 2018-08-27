#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/dynamic_routing_solver.prototxt -gpu 1 $@ 
