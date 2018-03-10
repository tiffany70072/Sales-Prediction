# 01/05/2018
# Method: matrix factorization with different time settings
# Original file name: mf_0112
# Goal: predict sales amount and sales item count (number)
# Data size: device = 499, product = 433, month = 24, week = 103

import tensorflow as tf
import numpy as np 
import pandas as pd
import random
import math

# restore model
from tensorflow.python.framework import ops
ops.reset_default_graph()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config = config))

class MatrixFactorization(object):
	def __init__(self, target = "amount", testing_file = None, 
				interval = "month", test_id = None, 
				lr = 0.01, lr_stage = 0.001, 
				epoch_stage = 30000, total_epoch = 60000, 
				lambda1 = 0.01, lambda3 = 1, 
				core_size = 16, time_setting = 2, alpha = None, 
				bias_setting = True, price_setting = False, 
				record_name = None, store_model = False, reuse_model = False, 
				combination = False):
		# load data
		self.interval = interval
		self.test_id = test_id # target testing time id
		self.target = target
		if self.interval == "month": self.load("data-112017/")
		elif self.interval == "week": self.load("data-122017-week/")

		# set target
		if target == "amount":
			self.real = self.amount_tensor
		elif target == "number":
			self.real = self.number_tensor 
			lambda2 = 0.1
		else: 
			print "Wrong target setting"
			exit()
		#if cross == True:
		#	dp_setting = True
		#	dt_setting = True
		#	pt_setting = True

		# set parameters
		self.lr  = lr # learning rate in the first stage
		self.lr2 = lr_stage # learning rate in the second stage
		self.epoch_stage = epoch_stage # modify learning rate here
		self.total_epoch = total_epoch # number of training epoch
		
		self.lamb1 = lambda1 # coefficient for weight regularization
		#self.lamb2 = lambda2 # coefficient for temporal regularization (x_t - x_(t-1))
		self.lamb3 = lambda3 # coefficient for non negative prediction
		self.time_setting = time_setting # setting for temporal weight
		self.alpha = alpha
		
		self.core_size    = core_size 
		#self.core_shape   = [self.core_size, self.core_size, self.core_size]
		#self.core_setting = core_setting # I: , core: tucker
		
		# feature settings
		self.bias_setting   = bias_setting # add bias on device, product, and global bias
		self.price_setting  = price_setting # consider the price of products
		#self.before_setting = before_setting # consider whether a product has sold before
		
		#self.dp_setting = dp_setting # cross term
		#self.dt_setting = dt_setting
		#self.pt_setting = pt_setting
		
		self.record_name = record_name # output results in this file
		self.store_model = store_model
		self.reuse_model = reuse_model
		
		# set file name
		if self.time_setting == 1:
			self.fout = open('record_mf' + str(self.time_setting) + "_" + str(self.record_name), 'w')
		else:
			self.fout = open('record_mf' + str(self.time_setting) + '-' + str(self.alpha) + "_" + str(self.record_name), 'w')
		self.fout.write("Record = " + str(self.record_name) + "\n")

		# early_stop
		self.min_epoch = 0
		self.min_MSE = 10000
		self.min_MAE = 10000
		self.min_ans = []
			
		# remove testing data from training data
		self.set_mask(testing_file)

		# training
		self.set_tensor()
		self.build_model()
		self.train()

		# output result
		fout = open('record_mf_0215', 'a')
		fout.write("week = %d, epoch = %d, MAE = %.3f, MSE = %.3f\n" %(test_id, self.min_epoch, self.min_MAE, self.min_MSE))
		fout.close()
		self.fout.close()
		if combination == True: self.output_combination()

	def load(self, path):
		print "\nLoading Data..."
		self.amount_tensor = np.load(path + "amount.npy")[:, :, :self.test_id+1] 	# data of sales amount
		self.number_tensor = np.load(path + "number.npy")[:, :, :self.test_id+1] 	# data of sales item count
		self.price_tensor  = np.load(path + "price.npy")[:, :, :self.test_id+1]  	# data of price
		self.init_mask = np.load(path + "exist.npy")[:, :, :self.test_id+1]	# is training data or not

		print "tensor.shape =", self.amount_tensor.shape 
		print "mask.shape   =", self.init_mask.shape
		print "price.shape  =", self.price_tensor.shape
		#print "amount =", self.amount_tensor[0, 0]

	def set_mask(self, testing_file):
		self.test_list = []
		zero = np.zeros([self.real.shape[2]])
		zero[-1] = 1
		self.mask = np.copy(self.init_mask) # initialize training data id

		# remove all training data in the "test" month
		for i in range(self.mask.shape[0]):
			for j in range(self.mask.shape[1]):
				if self.mask[i, j, self.test_id] == 1:
					if np.array_equal(self.mask[i, j, :], zero) == True:
						self.test_list.append([i, j])
					self.mask[i, j, self.test_id] = 0

		self.test_list = np.array(self.test_list)
		print "test.len() =", self.test_list.shape

		# 3d mask to 2d mask	
		temp = np.sum(self.mask, axis = 2)
		new_mask = np.zeros(temp.shape, dtype = float)
		for i in range(self.real.shape[0]):
			for j in range(self.real.shape[1]):
				if temp[i][j] != 0: new_mask[i][j] = 1
		print "#(Data) =", np.sum(new_mask), "/", self.real.shape[0]*self.real.shape[1]
		self.mask = new_mask

		self.price_tensor = np.sum(np.multiply(self.price_tensor, self.init_mask), axis = 2)\
									/np.sum(self.init_mask, axis = 2)
		
		'''# remove training data if it is testing data in the future
		for i in range(len(self.test_list)):
			for k in range(test_month):
				self.mask[self.test_list[i][0], self.test_list[i][1], k] = 0'''

	def set_tensor(self):
  		print "\nSetting parameter..."
  		
  		self.matrix1 = tf.Variable(tf.random_normal([self.real.shape[0], self.core_size])) # device property
  		self.matrix2 = tf.Variable(tf.random_normal([self.real.shape[1], self.core_size])) # product property
  		
  		self.global_bias =  tf.Variable(tf.random_normal([1]), name = "global_bias")
  		self.bias1 = tf.Variable(tf.random_normal([self.real.shape[0], 1])) # bias for device
  		self.bias2 = tf.Variable(tf.random_normal([1, self.real.shape[1]])) # bias for product
  		self.w = tf.Variable(tf.random_normal([1]), name = "weight_for_price")
  		self.w_b = tf.Variable(tf.random_normal([1]), name = "weight_for_exist_before")

  		self.real_nn  = tf.placeholder(tf.float32, [self.real.shape[0], self.real.shape[1]])
  		self.price_nn = tf.placeholder(tf.float32, [self.real.shape[0], self.real.shape[1]])
		#self.mask_nn  = tf.placeholder(tf.float32, self.mask.shape)
		#self.before_nn = tf.placeholder(tf.float32, self.real.shape)

	def build_model(self):
		def average():
			new_real = np.zeros([self.real.shape[0], self.real.shape[1]], dtype = float)
			for i in range(self.real.shape[0]):
				for j in range(self.real.shape[1]):
					if np.sum(self.init_mask[i][j]) != 0:
						new_real[i][j] = np.sum(self.init_mask[i][j] * self.real[i][j])/np.sum(self.init_mask[i][j])
					if i % 100 == 0 and j % 100 == 0: print i, j, new_real[i][j], self.real[i][j][50:]
			return new_real

		def exponential_moving_average():
			alpha_array = np.zeros(self.test_id, dtype = float)
			self.alpha = 0.5
			#self.alpha = tf.Variable(tf.random_normal([1])) # time coefficient for setting 2~4
  		
			alpha_array[self.test_id-1] = self.alpha
			for k in range(self.test_id-2, -1, -1): alpha_array[k] = alpha_array[k+1] * (1-self.alpha) 
				
			new_real = np.zeros([self.real.shape[0], self.real.shape[1]], dtype = float)
			for i in range(self.real.shape[0]):
				for j in range(self.real.shape[1]):
					new_real[i][j] = np.sum((self.real[i][j][:self.test_id] * self.init_mask[i][j][:self.test_id]) * alpha_array)
					#if np.sum(self.init_mask[i][j]) != 0: print new_real[i][j], self.real[i][j]
				#exit()
			return new_real

		def weighted_average_3():
			# exp[-alpha * (delta_t)], alpha should > 0, can be larger than 1
			alpha_array = np.zeros(self.test_id, dtype = float) 
			w = 1
			for k in range(self.test_id-1, -1, -1): 
				alpha_array[k] = w
				w *= math.exp(-self.alpha)
				if w < 0.001: break
				
			new_real = np.zeros([self.real.shape[0], self.real.shape[1]], dtype = float)
			for i in range(self.real.shape[0]):
				for j in range(self.real.shape[1]):
					if np.sum(self.init_mask[i][j][:self.test_id] * alpha_array) != 0:
						new_real[i][j] = np.sum((self.real[i][j][:self.test_id] * self.init_mask[i][j][:self.test_id])* alpha_array)
						new_real[i][j] /= np.sum(self.init_mask[i][j][:self.test_id] * alpha_array)
					else:
						self.mask[i][j] = 0
			return new_real

		def weighted_average_4():
			# (delta_t)^(-alpha), alpha should be positive and can be larger than 1
			alpha_array = np.zeros(self.test_id, dtype = float) 
			for k in range(self.test_id-1, -1, -1): 
				w = (self.test_id - k)**(-self.alpha)
				if w < 0.001: break
				alpha_array[k] = w

			new_real = np.zeros([self.real.shape[0], self.real.shape[1]], dtype = float)
			for i in range(self.real.shape[0]):
				for j in range(self.real.shape[1]):
					if np.sum(self.init_mask[i][j][:self.test_id] * alpha_array) != 0:
						new_real[i][j] = np.sum((self.real[i][j][:self.test_id] * self.init_mask[i][j][:self.test_id])* alpha_array)
						new_real[i][j] /= np.sum(self.init_mask[i][j][:self.test_id] * alpha_array)
					else:
						self.mask[i][j] = 0
			return new_real

		print "\nBuilding Model..."
		
		self.pred = tf.einsum('im,jm->ij', self.matrix1, self.matrix2)
		regularization = tf.reduce_sum(tf.square(self.matrix1)) + tf.reduce_sum(tf.square(self.matrix2))
		
		# add biases
		if self.bias_setting == True: 
			self.pred = self.pred + self.global_bias + self.bias1 + self.bias2 
			regularization += tf.square(self.global_bias) \
				+ tf.reduce_sum(tf.square(self.bias1)) + tf.reduce_sum(tf.square(self.bias2))

		# add feature: price
		#if self.price_setting == True:
		#	self.pred += self.w * self.price_nn #+ self.w2*tf.square(self.price_nn)
		#	regularization += self.w
		
		# add feature: exist_before
		#if self.before_setting == True:
		#	self.pred += self.w_b * self.before_nn
		#	regularization += self.w_b

		# add time regularzation and non-negative constraint
		regularization *= self.lamb1
		non_nagative = self.lamb3 * tf.reduce_sum(self.mask*(tf.abs(self.pred) - self.pred))

		# construct time settings
		if self.time_setting == 1: self.real_2d = average()
		elif self.time_setting == 2: self.real_2d = exponential_moving_average()
		elif self.time_setting == 3: self.real_2d = weighted_average_3()
		elif self.time_setting == 4: self.real_2d = weighted_average_4()
		
		# loss and error (for output)
		count = tf.reduce_sum(self.mask)
		loss = tf.reduce_sum(tf.square(self.mask*tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32) # real mean
		self.square_error = tf.reduce_sum(tf.square(self.mask * tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32)
		self.abs_error    = tf.reduce_sum(tf.abs   (self.mask * tf.subtract(self.real_nn, self.pred)))/tf.cast(count, tf.float32)
		
		self.loss = loss + regularization + non_nagative # loss for training
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		if self.store_model == True or self.reuse_model == True: 
			self.saver = tf.train.Saver()

	def output_loss(self):
		loss = self.sess.run(self.loss, 
			feed_dict = {self.real_nn: self.real_2d, self.price_nn: self.price_tensor})
		 
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
			if self.epoch % 200 == 0: self.output_loss()
			if self.epoch % 200 == 0: 
				self.output_train_error()
				self.output_test_error()

			self.sess.run(self.train_step, 
				feed_dict = {self.real_nn: self.real_2d, self.price_nn: self.price_tensor})
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
		#self.output_prediction()
		if self.store_model == True:
			save_path = self.saver.save(self.sess, 
				'model/model_' + self.record_name + '')  
				#global_step = self.total_epoch)
			print("Model saved in file: %s" % save_path)

		print "\nSetting # =", self.record_name
		print "Finished!\n"

	def output_train_error(self):
		square, abso = self.sess.run([self.square_error, self.abs_error], 
			feed_dict = {self.real_nn: self.real_2d, self.price_nn: self.price_tensor})
		 
		print "epoch = %d | train_error | square = %.3f, abs = %.3f" %(self.epoch, square, abso)
		self.fout.write("epoch = " + str(self.epoch) + " | train_error | ") 
		self.fout.write("square = %.3f, abs = %.3f\n" % (square, abso))

	def output_test_error(self, output = False):
		pred = self.sess.run(self.pred, 
			feed_dict = {self.real_nn: self.real_2d, self.price_nn: self.price_tensor})
		square = 0
		abso = 0
		
		for n in range(self.test_list.shape[0]):
			(i, j) = self.test_list[n][0:2]
			square += np.square(self.real[i][j][self.test_id] - pred[i][j])
			abso += abs(self.real[i][j][self.test_id] - pred[i][j])
		square /= float(len(self.test_list))
		abso /= float(len(self.test_list))
		
		print "epoch = %d | test_error | square = %.3f, abs = %.3f" %(self.epoch, square, abso)
		#print "test_num = %d" % len(self.test_list)
		self.fout.write("epoch = " + str(self.epoch) + " | test_error | ") 
		self.fout.write("square = %.3f, abs = %.3f\n" % (square, abso))

		if abso < self.min_MAE:
			self.min_epoch = self.epoch
			self.min_MAE = abso
			self.min_MSE = square
			self.min_ans = np.empty([len(self.test_list), 3], dtype = float)
			for n in range(self.test_list.shape[0]):
				(i, j) = self.test_list[n][0:2]
				self.min_ans[n][0:2] = self.test_list[n][0:2]
				self.min_ans[n][2] = pred[i][j]

			#for [i, j, k] in self.test_list: self.min_ans.append(pred[i][j][k])
			print "NEW: Epoch = %d, MAE = %.2f, MSE = %.2f" %(self.epoch, self.min_MAE, self.min_MSE)
			self.fout.write("NEW: Epoch = %d, MAE = %.2f, MSE = %.2f\n" %(self.epoch, self.min_MAE, self.min_MSE))

	def output_test_pred(self):
		self.fout.write("pred, real, abs_error, squ_error \n") 
		pred = self.sess.run(self.pred, 
			feed_dict = {self.real_nn: self.real_2d, self.price_nn: self.price_tensor})
		for [i, j] in self.test_list:
			self.fout.write("%.3f, %.3f, " %(pred[i][j], self.real[i][j][self.test_id])) 
			self.fout.write("%.3f, " % abs(pred[i][j] - self.real[i][j][self.test_id]))
			self.fout.write("%.3f\n" % (pred[i][j] - self.real[i][j][self.test_id])**2) 

	def output_combination(self):
		# store the results of tensor factorization, for next step: combination
		# result.shape = [len(testing data), 4]
		# col name = device id, product id, month (default is all the same), value
		print "Save the results for combination"
		print "#(Testing data) =", len(self.test_list)

		np.save("combination/combin_mf_" + self.record_name, self.min_ans)

def main():
	fout = open('record_mf_0215', 'a')
	fout.write("\nMF, method = 3, alpha = 1.0\n")
	fout.close()
	for test_id in range(50, 103): 
		MatrixFactorization(target = "amount", interval = "week", test_id = test_id,
			time_setting = 3, alpha = 1.0, 
			record_name = "week_" + str(test_id), lr = 0.01, epoch_stage = 10000, 
			reuse_model = False, store_model = False, total_epoch = 20000, combination = False)
	
	fout = open('record_mf_0215', 'a')
	fout.write("\nMF, method = 4, alpha = 2.0\n")
	fout.close()
	for test_id in range(50, 103): 
		MatrixFactorization(target = "amount", interval = "week", test_id = test_id,
			time_setting = 4, alpha = 2.0,  
			record_name = "week_" + str(test_id), lr = 0.01, epoch_stage = 10000, 
			reuse_model = False, store_model = False, total_epoch = 20000, combination = False)

	exit()	
	for test_id in range(50, 103): 
		MatrixFactorization(target = "amount", interval = "week", test_id = test_id,
			time_setting = 2, alpha = 0.3,  
			record_name = "week_" + str(test_id), lr = 0.01, epoch_stage = 10000, 
			reuse_model = False, store_model = False, total_epoch = 20000, combination = True)
	for test_id in [54, 55, 56, 57, 58]: 
		MatrixFactorization(target = "amount", interval = "week", test_id = test_id,
			time_setting = 2, alpha = 0.2,  
			record_name = "week_" + str(test_id), lr = 0.01, epoch_stage = 10000, 
			reuse_model = False, store_model = False, total_epoch = 20000, combination = True)
		# model = ...
	# [28, 37, 29, 30, 31, 18, 19, 25, 74,/ 92] V
	# [44, 45, 46, 54, 55, 56, /57, 58, 10, 12, 14, 15] V
	# [57, 58, 10, 12, 14, 15] V
	# [92, 13, 24, 88, /89, 90] V
	# [11, 16, 17, 20, 21, 22, 23, 26, 27, 32, 33, 34, 35,/] V
	# [47, 48, 49, 50, 51, 52] V
	# [71, 72, 73, 75, /] V
	# [76, 77, 78,/] V
	# --- 51 ---
	# [53, 59, 60, 61, 62, /] V
	# [63, 64, 65, 66, /] V
	# [71, 72, 73, 75, /] V
	# [76, 77, 78,/] V
	# [79, 80, 81, /] V
	# [82, 83, 84, /] V
	# [85, 86, 87, 89, 90, 91] now
	# [67, 68, 69, 70, 38, 39, 40, 41, 42, 43] V
	# [93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
	
if __name__ == '__main__':
	main()


		
