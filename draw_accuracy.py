import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

caffe.set_mode_gpu()


def draw_accuracy_hist(args):
	
	# Count cache
	count = 0
	correct = 0

	# Required Paths
	deploy_prototxt_file_path = args.prototxt_path
	caffe_model_file_path = args.caffemodel_path 
	test_lmdb_path = args.lmdb_path
	save_path = args.save_path


	# Loading trained model
	net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

	# Open lmdb test file
	lmdb_env = lmdb.open(test_lmdb_path)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()

	# Create count cache
	correct_count_dict = { i : 0 for i in list(range(0, 43))}
	total_count_dict = { i : 0 for i in list(range(0, 43))}


	for key, value in lmdb_cursor:
		# Get GT Label
		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)
		label = int(datum.label)

		# Predict test files
		out = net.forward()
		plabel = int(out['prob'][0].argmax(axis=0))
		count += 1
		iscorrect = label == plabel

		if label not in correct_count_dict.keys():
			correct_count_dict[label] = 0

		if label not in total_count_dict.keys():
			total_count_dict[label] = 0

		if iscorrect:
			correct_count_dict[label] += 1
			correct += 1

		total_count_dict[label] += 1


		if not iscorrect:
			sys.stdout.write("\r Error: key = %s, expected %i but predicted %i" % (key, label, plabel))
			sys.stdout.write("\r Accuracy: %.1f%%" % (100.*correct/count))
			sys.stdout.flush()

	for key, value in correct_count_dict.items():
		print (key, ":", value)

	for key, value in total_count_dict.items():
		correct_count_dict[key] = float(correct_count_dict[key]) / float(total_count_dict[key])

	for key, value in correct_count_dict.items():
		print (key, ":", value)

	print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")


	plt.bar(correct_count_dict.keys(), correct_count_dict.values(), 1 , color='b', edgecolor = 'r', linewidth = 0.2)
	plt.title('Accuracy by class')
	plt.savefig(save_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--prototxt_path', default='./models/caffenet/test.prototxt')
	# parser.add_argument('-caffemodel_path', default='./data/trained_models/caffenet/solver_iter_20000.caffemodel')
	# parser.add_argument('--save_path', default = './models/caffenet/plots/accuracy_by_class.png')
	# parser.add_argument('--lmdb_path', default = './data/lmdb_test/')
	# parser.add_argument('--model', default = 'caffenet')
	parser.add_argument('--prototxt_path', default='./models/alexnet/test.prototxt')
	parser.add_argument('-caffemodel_path', default='./data/trained_models/alexnet/solver_iter_80000.caffemodel')
	parser.add_argument('--save_path', default = './models/alexnet/plots/accuracy_by_class.png')
	parser.add_argument('--lmdb_path', default = './data/lmdb_test/')
	parser.add_argument('--model', default = 'alexnet')
	args = parser.parse_args()
	draw_accuracy_hist(args)