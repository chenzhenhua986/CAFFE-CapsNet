#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/capsule_solver.prototxt -gpu 1
