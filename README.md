# [RE510 | Lab3] GTSRB image classification - README
## Result Summary
| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ |:---------------:| -----:|
| col 3 is      | some wordy text | $1600 |
| col 2 is      | centered        |   $12 |
| zebra stripes | are neat        |    $1 |

## I. Required python package for my python codes
* `python==2.7`
* `pycaffe`
* `lmdb`
* `pandas`

## II. Store & Preprocess data

### Make LMDB
1. Store data in `<./data/train>` and `<./data/test>`
2. Make `path_test.txt` and `path_label_train.txt`
  * Run `$ python gen_lmdb.py'
3. Store LMDB files in `./data/lmdb_train` and `./data/lmdb_test`
  * Run `$ run_convert_imageset_to_lmdb_test.sh`  
  * Run `$ run_convert_imageset_to_lmdb_train.sh`

### Make meanfiles
1. Create mean files: `./data/lmdb_train/gtsrb_train_mean.binaryproto` and `./data/lmdb_test/gtsrb_test_mean.binaryproto'
  * Run `$ make_gtsrb_mean_test.sh`
  * Run `$ make_gtsrb_mean_train.sh`

## III. Train (CaffeNet as an example)
1. CaffeNet 
  * Run `$ sh train_caffenet.sh`
2. AlexNet
  * Run `$ sh train_alexnet.sh`
3. CaffeNet + BatchNormalization
  * Run `$ sh train_shwangnet.sh`
4. AlexNet + BatchNormalization
  * Run `$ sh train_alexnetbn.sh`
5. CaffeNet + BatchNormalization + Dropout (0.2)
  * Run `$ sh train_shwangdrop.sh`
