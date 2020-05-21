./caffe/build/tools/caffe train --solver=./models/alexnet/solver.prototxt 2>&1 | tee ./models/alexnet/logs/alexnet.001_weight_decay_0.005.log
