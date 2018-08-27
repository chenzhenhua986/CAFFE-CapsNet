source ~/.bashrc
rm -rf f
export PYTHONPATH=$PYTHONPATH:pwd
/l/vision/v5/chen478/Spring2018/caffe/build/tools/extract_features.bin /l/vision/v5/chen478/Spring2018/caffe/examples/mnist/capsule_new_iter_3000.caffemodel /l/vision/v5/chen478/Spring2018/caffe/examples/mnist/deploy.prototxt accuracy f 1 lmdb GPU 3
echo 'done'
