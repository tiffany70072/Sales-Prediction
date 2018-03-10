# 12/02/2017
# Method: tensor factorization
# Original file name: tf_1202
# Goal: predict sales amount and sales item count (number)
# Data size: device = 499, product = 433, month = 24
# add settings and library 

import tensorflow as tf
import numpy as np 
import pandas as pd
import random

# restore model
from tensorflow.python.framework import ops
ops.reset_default_graph()
#from ops_1202 import *

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config = config))

class TensorFactorization(object):
	def __init__(self, target = "amount", testing_file = None, 
				interval = "month", time_id = None, 
				lr = 0.01, lr_stage = 0.001, 
				epoch_stage = 30000, total_epoch = 60000, 
				lambda1 = 0.01, lambda2 = 0.003, lambda3 = 1, 
				core_size = 16, core_setting = "I",
				bias_setting = True, price_setting = False, before_setting = False, 
				dp_setting = False, dt_setting = False, pt_setting = False, cross = False,
				record_name = None, store_model = False, reuse_model = False, 
				combination = False):
		# load data
		self.interval = interval
		self.time_id = time_id # target testing time id
		if self.interval == "month": self.load("data-112017/")
		elif self.interval == "week": self.load("data-122017-week")

		# set target
		if target == "amount":
			self.real = self.amount_tensor
		elif target == "number":
			self.real = self.number_tensor 
			lambda2 = 0.1
		else: 
			print "Wrong target setting"
			exit()
		if cross == True:
			dp_setting = True
			dt_setting = True
			pt_setting = True

		# set parameters
		self.lr  = lr # learning rate in the first stage
		self.lr2 = lr_stage # learning rate in the second stage
		self.epoch_stage = epoch_stage # modify learning rate here
		self.total_epoch = total_epoch # number of training epoch
		
		self.lamb1 = lambda1 # coefficient for weight regularization
		self.lamb2 = lambda2 # coefficient for temporal regularization (x_t - x_(t-1))
		self.lamb3 = lambda3 # coefficient for non negative prediction
		
		self.core_size    = core_size 
		self.core_shape   = [self.core_size, self.core_size, self.core_size]
		self.core_setting = core_setting # I: , core: tucker
		
		# feature settings
		self.bias_setting   = bias_setting # add bias on device, product, and global bias
		self.price_setting  = price_setting # consider the price of products
		self.before_setting = before_setting # consider whether a product has sold before
		
		self.dp_setting = dp_setting # cross term
		self.dt_setting = dt_setting
		self.pt_setting = pt_setting
		
		self.record_name = record_name # output results in this file
		self.store_model = store_model
		self.reuse_model = reuse_model
		
		# set file name
		self.fout = open('record_' + str(self.record_name), 'w')
		self.fout.write("Record = " + str(self.record_name) + "\n")
			
		# remove testing data from training data
		self.set_mask(testing_file)

		# training
		self.set_tensor()
		self.build_model()
		self.train()

		# output result
		self.fout.close()
		if combination == True: self.output_combination()

	def load(self, path):
		print "\nLoading Data..."
		self.amount_tensor = np.load(path + "amount.npy") 	# data of sales amount
		self.number_tensor = np.load(path + "number.npy") 	# data of sales item count
		self.price_tensor  = np.load(path + "price.npy")  	# data of price
		self.init_mask = np.load(path + "exist.npy")		# is training data or not

		print "tensor.shape =", self.amount_tensor.shape 
		print "mask.shape   =", self.init_mask.shape
		print "price.shape  =", self.price_tensor.shape
		print "amount =", self.amount_tensor[0, 0]

	def set_mask(self, testing_file):
		self.test_list = np.load("data-112017/test_" + testing_file + ".npy")
		self.mask = np.copy(self.init_mask) # initialize training data id
		test_month = 23

		# remove all training data in the "test" month
		for i in range(self.mask.shape[0]):
			for j in range(self.mask.shape[1]):
				if self.mask[i, j, test_month] == 1:
					self.mask[i, j, test_month] = 0
		# remove training data if it is testing data in the future
		for i in range(len(self.test_list)):
			for k in range(test_month):
				self.mask[self.test_list[i][0], self.test_list[i][1], k] = 0

	def set_tensor(self):
  		print "\nSetting parameter..."

  		if self.core_setting == "core":
  			self.core = tf.Variable(tf.random_normal(self.core_shape))
  		elif self.core_setting == "I":
  			self.core = np.zeros(self.core_shape, dtype = float)
  			for i in range(self.core_size):	self.core[i][i][i] = 1.0
  			self.core = tf.cast(self.core, tf.float32)
  		else: 
  			print "Wrong core setting!"
  			exit()
  		
  		self.matrix1 = tf.Variable(tf.random_normal([self.real.shape[0], self.core_size])) # device property
  		self.matrix2 = tf.Variable(tf.random_normal([self.real.shape[1], self.core_size])) # product property
  		self.matrix3 = tf.Variable(tf.random_normal([self.real.shape[2], self.core_size])) # time property
  		self.global_bias =  tf.Variable(tf.random_normal([1]), name = "global_bias")
  		self.bias1 = tf.Variable(tf.random_normal([self.real.shape[0], 1, 1])) # bias for device
  		self.bias2 = tf.Variable(tf.random_normal([1, self.real.shape[1], 1])) # bias for product
  		self.w = tf.Variable(tf.random_normal([1]), name = "weight_for_price")
  		#self.w2 = tf.Variable(tf.random_normal([1]), name = "weight_for_price")
  		self.w_b = tf.Variable(tf.random_normal([1]), name = "weight_for_exist_before")

  		self.real_nn  = tf.placeholder(tf.float32, self.real.shape)
  		self.price_nn = tf.placeholder(tf.float32, self.real.shape)
		self.mask_nn  = tf.placeholder(tf.float32, self.mask.shape)
		self.before_nn = tf.placeholder(tf.float32, self.real.shape)

		print "#(Variables) =",
		print self.core_size**3 + (self.real.shape[0]+self.real.shape[1]+self.real.shape[2])*self.core_size

	def build_model(self):
		print "\nBuilding Model..."
		
		self.pred = tf.einsum('mnl,im,jn,kl->ijk', self.core, self.matrix1, self.matrix2, self.matrix3)

		# add cross term
		if self.dp_setting == True: 
			bias_dp = tf.einsum('in,jn->ij', self.matrix1, self.matrix2)
			self.pred += tf.expand_dims(bias_dp, 2)
		if self.dt_setting == True: 
			bias_dt = tf.einsum('in,jn->ij', self.matrix1, self.matrix3)
			self.pred += tf.expand_dims(bias_dt, 1)
		if self.pt_setting == True: 
			bias_pt = tf.einsum('in,jn->ij', self.matrix2, self.matrix3)
			self.pred += tf.expand_dims(bias_pt, 0)
		
		regularization = tf.reduce_sum(tf.square(self.matrix1)) \
			+ tf.reduce_sum(tf.square(self.matrix2)) \
			+ tf.reduce_sum(tf.square(self.matrix3))
		
		# add biases
		if self.bias_setting == True: 
			self.pred = self.pred + self.global_bias + self.bias1 + self.bias2 
			regularization += tf.square(self.global_bias) \
				+ tf.reduce_sum(tf.square(self.bias1)) + tf.reduce_sum(tf.square(self.bias2))

		# add feature: price
		if self.price_setting == True:
			self.pred += self.w * self.price_nn #+ self.w2*tf.square(self.price_nn)
			regularization += self.w
		
		# add feature: exist_before
		if self.before_setting == True:
			self.pred += self.w_b * self.before_nn
			regularization += self.w_b

		# add time regularzation and non-negative constraint
		regularization *= self.lamb1
		temporal_regul = self.lamb2 * tf.reduce_sum(tf.square(self.matrix3[:-1] - self.matrix3[1:])) # for tesing month = 23
		non_nagative = self.lamb3 * tf.reduce_sum(self.mask*(tf.abs(self.pred) - self.pred))
		
		# loss and error (for output)
		count = tf.reduce_sum(self.mask)
		loss = tf.reduce_sum(tf.square(self.mask*tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32) # real mean
		self.square_error = tf.reduce_sum(tf.square(self.mask * tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32)
		self.abs_error    = tf.reduce_sum(tf.abs   (self.mask * tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32)
		
		self.loss = loss + regularization + temporal_regul + non_nagative # loss for training
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		if self.store_model == True or self.reuse_model == True: 
			self.saver = tf.train.Saver()

	def output_loss(self):
		loss = self.sess.run(self.loss, 
			feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
		 
		'''if self.flag == 1 and (loss < 50 or self.epoch > 20000):
			self.lr *= 0.1
			print "learning rate =", self.lr
			self.fout.write("learning rate =" + str(self.lr))
			self.flag = 0 '''

		print "epoch = %d | loss = %.3f" %(self.epoch, loss)
		self.fout.write("epoch = " + str(self.epoch) + " | ") 
		self.fout.write("%.3f\n" % (loss))

	def train(self):
		print "\nStart training..."
		print "Setting # =", self.record_name, "\n"
		self.init = tf.global_variables_initializer()
		self.sess = tf.Session() #self.sess = tf.InteractiveSession()
		self.sess.run(self.init)

		if self.reuse_model == True:
			self.saver = tf.train.import_meta_graph("model/model_" + str(self.record_name) + ".meta")
			self.saver.restore(self.sess, 'model/model_' + self.record_name + '')
		 	print("Model restored.")
		 	'''with tf.Session() as sess:
				self.saver = tf.train.import_meta_graph('model/model_' + str(self.record_name) + '.meta')
	  			self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))'''

		self.epoch = 0
		while self.epoch < self.total_epoch:
			if self.epoch % 500 == 0: self.output_loss()
			if self.epoch % 1000 == 0: 
				self.output_train_error()
				self.output_test_error()

			self.sess.run(self.train_step, 
				feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
			self.epoch += 1

			if self.epoch == self.epoch_stage: 
				self.lr = self.lr2
				self.fout.write("-learning rate = %f-\n" %self.lr)

			if self.store_model == True and self.epoch % 10000 == 0 and self.epoch != 0: 
				save_path = self.saver.save(self.sess, 
					'model/model_' + str(self.record_name) + '')
				print("Model saved in file: %s" % save_path)
			
		self.output_loss()
		self.output_train_error()
		self.output_test_error()
		self.output_test_pred()
		if self.store_model == True:
			save_path = self.saver.save(self.sess, 
				'model/model_' + self.record_name + '')  
				#global_step = self.total_epoch)
			print("Model saved in file: %s" % save_path)

		print "\nSetting # =", self.record_name
		print "Finished!\n"

	def output_train_error(self):
		square, abso = self.sess.run([self.square_error, self.abs_error], 
			feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
		 
		print "epoch = %d | train_error | square = %.3f, abs = %.3f" %(self.epoch, square, abso)
		self.fout.write("epoch = " + str(self.epoch) + " | train_error | ") 
		self.fout.write("square = %.3f, abs = %.3f\n" % (square, abso))

	def output_test_error(self, output = False):
		pred = self.sess.run(self.pred, 
			feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
		square = 0
		abso = 0
		
		for [i, j, k] in self.test_list:
			square += np.square(self.real[i][j][k] - pred[i][j][k])
			abso += abs(self.real[i][j][k] - pred[i][j][k])
		square /= float(len(self.test_list))
		abso /= float(len(self.test_list))
		
		print "epoch = %d | test_error | square = %.3f, abs = %.3f" %(self.epoch, square, abso)
		#print "test_num = %d" % len(self.test_list)
		self.fout.write("epoch = " + str(self.epoch) + " | test_error | ") 
		self.fout.write("square = %.3f, abs = %.3f\n" % (square, abso))

	def output_test_pred(self):
		self.fout.write("pred, real, abs_error, squ_error \n") 
		pred = self.sess.run(self.pred, 
			feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
		for [i, j, k] in self.test_list:
			self.fout.write("%.3f, %.3f, " %(pred[i][j][k], self.real[i][j][k])) 
			self.fout.write("%.3f, " % abs(pred[i][j][k] - self.real[i][j][k]))
			self.fout.write("%.3f\n" % (pred[i][j][k] - self.real[i][j][k])**2) 

	def output_combination(self):
		# store the results of tensor factorization, for next step: combination
		# result.shape = [len(testing data), 4]
		# col name = device id, product id, month (default is all the same), value
		print "Save the results for combination"
		print "#(Testing data) =", len(self.test_list)

		result = np.empty([len(self.test_list), 4], dtype = float)
		pred = self.sess.run(self.pred, 
			feed_dict = {self.real_nn: self.real, self.price_nn: self.price_tensor})
		for i in range(self.test_list.shape[0]):
			result[i][0:3] = self.test_list[i][0:3]
			result[i][3] = pred[self.test_list[i][0]][self.test_list[i][1]][self.test_list[i][2]]
			print result[i]

		np.save("combination/combin_tf_" + self.record_name, result)

def main():
	model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-9(test)", 
		reuse_model = False, store_model = True, total_epoch = 10, combination = True)
	#model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-9(test)", 
	#	reuse_model = False, store_model = True, total_epoch = 1)
	model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-9(test)", 
		reuse_model = True, store_model = True, total_epoch = 1)
	#model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-9(test)", 
	#	reuse_model = True, store_model = False, total_epoch = 0)
	
	'''
	model = TensorFactorization(target = "amount", testing_file = "23_1000_0", record_name = "1-5", 
		cross = False)
	model = TensorFactorization(target = "amount", testing_file = "23_1000_0", record_name = "1-6", 
		cross = True)
	model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-7", 
		cross = False)
	model = TensorFactorization(target = "amount", testing_file = "23_first", record_name = "1-8", 
		cross = True)
	'''

if __name__ == '__main__':
	main()


		
