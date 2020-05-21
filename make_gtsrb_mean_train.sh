#!/usr/bin/env sh
# Compute the mean image from the gtsrb training lmdb

LMDB_PATH=./data/lmdb_train
SAVE_PATH=./data/mean_lmdb_train
TOOLS=./caffe/build/tools

$TOOLS/compute_image_mean $LMDB_PATH\
  $SAVE_PATH/gtsrb_train_mean.binaryproto

echo "Done."
