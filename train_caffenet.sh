./caffe/build/tools/caffe train --solver=./models/caffenet/solver.prototxt 2>&1 | tee ./models/caffenet/logs/caffenet_lr_0.001_weight_decay_0.005.log
