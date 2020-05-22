import pandas as pd
from matplotlib import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from os.path import join
import argparse

def plot_stuff(log_path_train, log_path_test, save_dir, model):
	train_log = pd.read_csv(log_path_train)
	test_log = pd.read_csv(log_path_test)

	plt.figure(1)
	plt.plot(train_log["NumIters"][:15000], train_log["loss"][:15000], alpha=0.4)
	plt.title('TRAIN LOSS: '+model)
	plt.xlabel('iteration')
	plt.ylabel('train loss')
	plt.savefig(join(save_dir, 'train_loss.png'))

	plt.figure(2)
	plt.plot(test_log["NumIters"][:15000], test_log["loss"][:15000], 'g')
	plt.title('TEST LOSS: '+model)
	plt.xlabel('iteration')
	plt.ylabel('test loss')
	plt.savefig(join(save_dir, 'test_loss.png'))

	plt.figure(3)
	plt.plot(test_log["NumIters"][:15000], test_log["accuracy"][:15000], 'r')
	plt.title('TEST ACCURACY: '+model)
	plt.xlabel('iteration')
	plt.ylabel('test accuracy')
	plt.savefig(join(save_dir, 'test_accuracy.png'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--log_path_train', default='./models/caffenet/logs/train_caffenet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('-log_path_test', default='./models/caffenet/logs/test_caffenet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('--save_dir', default = './models/caffenet/plots')
	# parser.add_argument('--model', default = 'caffenet')
	# parser.add_argument('--log_path_train', default='./models/alexnet/logs/train_alexnet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('-log_path_test', default='./models/alexnet/logs/test_alexnet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('--save_dir', default = './models/alexnet/plots')
	# parser.add_argument('--model', default = 'alexnet')
	# parser.add_argument('--log_path_train', default='./models/shwangnet/logs/train_shwangnet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('-log_path_test', default='./models/shwangnet/logs/test_shwangnet_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('--save_dir', default = './models/shwangnet/plots')
	# parser.add_argument('--model', default = 'caffenet_BN')
	# parser.add_argument('--log_path_train', default='./models/alexnetbn/logs/train_alexnetbn_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('-log_path_test', default='./models/alexnetbn/logs/test_alexnetbn_lr_0.001_weight_decay_0.005.log')
	# parser.add_argument('--save_dir', default = './models/alexnetbn/plots')
	# parser.add_argument('--model', default = 'alexnet_BN')
	parser.add_argument('--log_path_train', default='./models/shwangdrop/logs/train_shwangnet_lr_0.001_weight_decay_0.005.log')
	parser.add_argument('-log_path_test', default='./models/shwangdrop/logs/test_shwangnet_lr_0.001_weight_decay_0.005.log')
	parser.add_argument('--save_dir', default = './models/shwangdrop/plots')
	parser.add_argument('--model', default = 'caffenet_BN_FCN_dropout_0.2_.png')
	args = parser.parse_args()

	plot_stuff(args.log_path_train, args.log_path_test, args.save_dir, args.model)