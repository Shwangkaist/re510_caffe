LOG_DIR=./models/alexnetbn/logs
LOG_NAME=alexnetbn_lr_0.001_weight_decay_0.005.log

python ./caffe/tools/extra/parse_log.py $LOG_DIR/$LOG_NAME $LOG_DIR
