import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

'''A convolutional neural network module to extract features from the images'''
class CNN(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.image_shape = [28,28,1];
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		print('Building the Embedding CNN......');

	def run(self,images,train,reuse=False):
		image_shape = self.image_shape;
		bn = self.batch_norm;

		with tf.variable_scope('cnn', reuse=reuse):
			conv1 = convolution(images,3,3,64,1,1,'conv1');
			conv1 = batch_norm(conv1,'bn1',train,bn);
			conv1 = nonlinear(conv1,'relu');
			pool1 = max_pool(conv1,2,2,2,2,);
			conv2 = convolution(pool1,3,3,64,1,1,'conv2');
			conv2 = batch_norm(conv2,'bn2',train,bn);
			conv2 = nonlinear(conv2,'relu');
			pool2 = max_pool(conv2,2,2,2,2);
			conv3 = convolution(pool2,3,3,64,1,1,'conv3');
			conv3 = batch_norm(conv3,'bn3',train,bn);
			conv3 = nonlinear(conv3,'relu');
			pool3 = max_pool(conv3,2,2,2,2);
			conv4 = convolution(pool3,3,3,64,1,1,'conv4',padding='VALID');
			conv4 = batch_norm(conv4,'bn4',train,bn);
			conv4 = nonlinear(conv4,'relu');
			pool4 = max_pool(conv4,2,2,2,2);
		
		output = tf.reshape(pool4,[self.batch_size,-1]);
		return output;

