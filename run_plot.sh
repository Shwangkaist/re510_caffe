CHART_TYPE="2"
IMG_SAVE_PATH=./models/caffenet/plots/test.png
TRAIN_LOG_PATH=./models/caffenet/logs/train_caffenet_lr_0.001_weight_decay_0.005_train.log
TEST_LOG_PATH=./models/caffenet/logs/test_caffenet_lr_0.001.weight_decay_0.005_train.log

python ./plot_training_log.py $CHART_TYPE $IMG_SAVE_PATH $TRAIN_LOG_PATH
