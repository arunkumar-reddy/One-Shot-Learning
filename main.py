import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';
sys.setrecursionlimit(10000);

import tensorflow as tf;
import argparse;
from model import *;
from dataset import *;

def main(argv):
	parser = argparse.ArgumentParser();
	'''Model Parameters'''
	parser.add_argument('--phase', default='train', help='Train or Test');
	parser.add_argument('--load', action='store_true', default=False, help='Load the trained model');
	parser.add_argument('--num_classes', type=int, default=5, help='Number of classes per support set');
	parser.add_argument('--num_samples', type=int, default=5, help='Number of samples per class');
	parser.add_argument('--rnn_units', type=int, default=3, help='Number of RNN units to use for encoder-decoder');
	parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size of the RNN unit');
	parser.add_argument('--train_dir', default='./train/', help='Directory containing the training images');
	parser.add_argument('--test_dir', default='./test/', help='Directory containing testing images');
	parser.add_argument('--save_dir', default='./models/', help='Directory to save the trained model');
	parser.add_argument('--save_period', type=int, default=2000, help='Period to save the trained model');
	'''Hyper parameters'''
	parser.add_argument('--solver', default='adam', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			#data = train_data(args);
			model = Model(args,'train');
			sess.run(tf.global_variables_initializer());
			if(args.load):
				model.load(sess);
			model.Train(sess,data);
		else:
			data = test_data(args);
			model = Model(args,'test');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.Test(sess,data);

main(sys.argv);