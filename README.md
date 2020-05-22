# [RE510 | Lab3] GTSRB image classification - README

## I. Required python package for my python codes
* `python`==2.7
* `pycaffe`
* `lmdb`
* `pandas'

## II. Store & Preprocess data

### Make LMDB
1. Store data in `<./data/train>` and `<./data/test>`
2. Make `path_test.txt` and `path_label_train.txt` by running `$ python gen_lmdb.py'
3. Run `$ run_convert_imageset_to_lmdb_test.sh` and `$ run_convert_imageset_to_lmdb_train.sh` to get lmdb files in `./data/lmdb_train` and `./data/lmdb_test`

### Make meanfiles
1. Run `$ make_gtsrb_mean_test.sh` and `$ make_gtsrb_mean_train.sh` to create mean files as `./data/lmdb_train/gtsrb_train_mean.binaryproto` and `./data/lmdb_test/gtsrb_test_mean.binaryproto'

## III. Train (CaffeNet as an example)
1. Run `$ sh train_caffenet.sh`
