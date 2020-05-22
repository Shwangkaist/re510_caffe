#!/usr/bin/env sh
# Compute the mean image from the gtsrb training lmdb

LMDB_PATH=./data/lmdb_test_kaist
SAVE_PATH=./data/mean_lmdb_test_kaist
TOOLS=./caffe/build/tools

$TOOLS/compute_image_mean $LMDB_PATH\
  $SAVE_PATH/gtsrb_test_kaist_mean.binaryproto

echo "Done."
