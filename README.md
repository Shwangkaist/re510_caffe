# [RE510] GTSRB image classification
## Test Accuracy Summary
> Test Log files stored in `./models/$modelname/logs/test_sthsth.log` for Verification

| CaffeNet | AlexNet | CaffeNet + BatchNorm | AlexNet + BatchNorm | CaffeNet + BatchNorm + Dropout: 0.2 |
| :---: | :---: | :---: | :---: | :---: |
| 87.6% | 89.8% | __90.0%__ | 88.5% | 88.6% |


## I. Required python package for my python codes
* `python==2.7`
* `pycaffe`
* `lmdb`
* `pandas`

## II. Store & Preprocess data

#### Make LMDB
1. Store data in `<./data/train>` and `<./data/test>`
2. Make `path_test.txt` and `path_label_train.txt`
    * `$ python gen_lmdb.py`
3. Store LMDB files in `./data/lmdb_train` and `./data/lmdb_test`
    * `$ sh run_convert_imageset_to_lmdb_test.sh`  
    * `$ sh run_convert_imageset_to_lmdb_train.sh`

#### Make meanfiles
1. Create mean files: `./data/lmdb_train/gtsrb_train_mean.binaryproto` and `./data/lmdb_test/gtsrb_test_mean.binaryproto`
    * `$ sh make_gtsrb_mean_test.sh`
    * `$ sh make_gtsrb_mean_train.sh`

## III. Train (CaffeNet as an example)
1. CaffeNet : `$ sh train_caffenet.sh`
2. AlexNet  : `$ sh train_alexnet.sh`
3. CaffeNet + BatchNormalization:`$ sh train_shwangnet.sh`
4. AlexNet + BatchNormalization: `$ sh train_alexnetbn.sh`
5. CaffeNet + BatchNormalization + Dropout (0.2): `$ sh train_shwangdrop.sh`

## IV. Post-processing
1. Parse log files (locaeted at `./models/$modelname/logs`) : `$ sh parse_$modelname_train_log.sh
2. Draw train loss, test loss, and test accuracy log: `$ python draw_train_log.py`
3. Draw test accuracy by class (__REPLACED MATCAFFE__): `$ python draw_accuracy.py`
