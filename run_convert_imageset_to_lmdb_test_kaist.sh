IMG_PATH=./data
TXT_PATH=./path_test_kaist.txt
LMDB_PATH=./data/lmdb_test_kaist
TOOL_PATH=./caffe/build/tools/convert_imageset

GLOG_logtostderr=1 $TOOL_PATH \
    --resize_height=200 --resize_width=200 --shuffle  \
    $IMG_PATH \
    $TXT_PATH \
    $LMDB_PATH
