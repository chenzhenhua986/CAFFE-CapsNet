import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import numpy as np
import matplotlib.image as mpimg

lmdb_env = lmdb.open('examples/mnist/f')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    feat = caffe.io.datum_to_array(datum)

    print 'Processing image: ' + str(feat)



