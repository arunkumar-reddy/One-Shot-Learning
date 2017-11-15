import os;
import math;
import numpy as np;
import tensorflow as tf;
import tensorflow.contrib.rnn as rnn;
from nn import *;

'''A Matching Neural Network to encode the images in context of the support set'''
class MNN(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.rnn_units = params.rnn_units;
		self.hidden_size = params.hidden_size;
		print('Building the Matching Network......');

	def run(self,inputs):
		with tf.variable_scope('encoder'):
			encoder_forward = [];
			encoder_backward = [];
			for i in range(self.rnn_units):
				forward_cell = rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				backward_cell = rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					forward_cell = rnn.DropoutWrapper(forward_cell,input_keep_prob=1.0,output_keep_prob=0.9);
					backward_cell = rnn.DropoutWrapper(backward_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				encoder_forward.append(forward_cell);
				encoder_backward.append(backward_cell);
			encoder_output,forward_state,backward_state = rnn.stack_bidirectional_rnn(encoder_forward,encoder_backward,inputs,dtype=tf.float32);

		with tf.variable_scope('decoder'):
			decoder_forward = [];
			decoder_backward = [];
			for i in range(self.rnn_units):
				forward_cell = rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				backward_cell = rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					forward_cell = rnn.DropoutWrapper(forward_cell,input_keep_prob=1.0,output_keep_prob=0.9);
					backward_cell = rnn.DropoutWrapper(backward_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				decoder_forward.append(forward_cell);
				decoder_backward.append(backward_cell);
			decoder_output,forward_state,backward_state = rnn.stack_bidirectional_rnn(decoder_forward,decoder_backward,encoder_output,forward_state,backward_state,dtype=tf.float32);
		
		output = decoder_output;
		states = (forward_state,backward_state);
		return output,states;