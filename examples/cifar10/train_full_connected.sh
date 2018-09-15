#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/full_connected_capsule_solver1.prototxt -gpu 1 $@

