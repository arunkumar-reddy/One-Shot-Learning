import tensorflow as tf;
import numpy as np;
import os;

from tqdm import tqdm;
from skimage.io import imread;
from dataset import *;
from cnn import *;
from mnn import *;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		self.image_shape = [28,28,1];
		self.num_classes = params.num_classes;
		self.num_samples = params.num_samples;
		self.support_size = params.num_classes*params.num_samples;
		self.save_dir = os.path.join(params.save_dir,self.params.solver+'/');
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.saver = tf.train.Saver(max_to_keep = 10);
		self.epsilon = 1e-9;
		self.build();

	def build(self):
		support_images = tf.placeholder(tf.float32,[self.batch_size,self.support_size]+self.image_shape);
		support_labels = tf.placeholder(tf.float32,[self.batch_size,self.support_size,self.num_classes]);
		images = tf.placeholder(tf.float32,[self.batch_size]+self.image_shape);
		labels = tf.placeholder(tf.float32,[self.batch_size,self.num_classes]);
		train = tf.placeholder(tf.bool);

		cnn = CNN(self.params,self.phase);
		mnn = MNN(self.params,self.phase);
		encoded_images = [];
		train_image = cnn.run(images,train);
		for image in tf.unstack(support_images,axis=1):
			support_image = cnn.run(image,train,reuse=True);
			encoded_images.append(support_image);
		encoded_images.append(train_image);

		encoded_images,states = mnn.run(encoded_images);
		encoded_images = tf.stack(encoded_images);
		supports = encoded_images[:-1];
		train_images = encoded_images[-1];

		'''Attention Mechanism by looking for similar encodings'''
		similarities = [];
		for support in tf.unstack(supports,axis=0):
			support_sum = tf.reduce_sum(tf.square(support),1,keep_dims=True);
			support_magnitude = tf.rsqrt(tf.clip_by_value(support_sum,self.epsilon,float("inf")));
			product = tf.matmul(tf.expand_dims(train_images,1),tf.expand_dims(support,2));
			product = tf.squeeze(product,[1,]);
			cosine_similarity = product*support_magnitude;
			similarities.append(cosine_similarity);

		similarities = tf.concat(similarities,axis=1);
		similarities = tf.nn.softmax(similarities);
		logits = tf.squeeze(tf.matmul(tf.expand_dims(similarities,1),support_labels));
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits);
		outputs = tf.argmax(logits,1);
		accuracy = tf.reduce_mean(tf.cast(tf.equal(outputs,tf.argmax(labels,1)),tf.float32));

		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		optimizer = solver.minimize(loss,global_step=self.global_step);

		self.support_images = support_images;
		self.support_labels = support_labels;
		self.images = images;
		self.labels = labels;
		self.train = train;
		self.loss = loss;
		self.outputs = outputs;
		self.accuracy = accuracy;
		self.optimizer = optimizer;
		print('Model built......');

	def Train(self,sess,data):
		pass;

	def Test(self,sess,data):
		pass;

	def save(self,sess):
		print(('Saving model to %s......'% self.save_dir));
		self.saver.save(sess,self.save_dir,self.generator_step);

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first...");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);