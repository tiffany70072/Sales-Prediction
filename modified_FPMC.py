# Method: FPMC with end to end (another is self-defined probability threshold)
# 01/23/2018
# Original file name: FPMC_value_0226
# Goal: predict sales out or not (binary classification)
# Data size: device = 450, product = 351, month = 24, week = 103

import tensorflow as tf
import numpy as np 
import pandas as pd
import random

# restore model
from tensorflow.python.framework import ops
ops.reset_default_graph()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config = config))

class FPMC(object):
	def __init__(self, target = "amount", time_id = None, interval = "month", 
				lr = 0.001, total_epoch = 60000, 
				lamb = 0.01, core_size = 16, train_all = False, 
				dl_setting = True, di_setting = True, li_setting = False,
				record_name = None, store_model = False, reuse_model = False):
		
		# load data
		self.interval = interval
		self.time_id = time_id 					# target testing time id
		self.target = target
		self.train_all = train_all
		if self.interval == "week": self.load("data-012018-week/")

		# parameters
		self.lr  = lr 							# learning rate 
		self.total_epoch = total_epoch 			# number of training epoch
		self.lamb = lamb 						# coefficient for weight regularization
		if self.train_all == True: self.bs = 500
		else: self.bs = 128
		self.k_di, self.k_id, self.k_il, self.k_li = core_size, core_size, core_size, core_size
		
		self.dl_setting = dl_setting 			# cross term in tensor
		self.di_setting = di_setting
		self.li_setting = li_setting
		
		
		self.store_model = store_model
		self.reuse_model = reuse_model

		# early_stop
		self.min_epoch = 0
		self.min_MSE = 10000
		self.min_MAE = 10000
		self.min_ans = []
		
		# set file name
		self.record_name = record_name 			# output results in this file
		self.fout = open('record_' + str(self.record_name), 'w')
		self.fout.write("Record = " + str(self.record_name) + "\n")
			
		# remove testing data from training data
		#self.set_mask(testing_file)
		self.preprocess()

		# training
		self.set_tensor()
		self.build_model()
		self.train()

		# output result
		fout = open('record_(value)_0211', 'a')
		fout.write("week = %d, epoch = %d, MAE = %.3f, MSE = %.3f\n" %(time_id, self.min_epoch, self.min_MAE, self.min_MSE))
		fout.close()
		self.fout.close()

	def load(self, path):
		print "\nLoading Data..."
		if self.target == "amount": self.real = np.load(path + "amount.npy")	# data of sales amount
		elif self.target == "number": self.real = np.load(path + "number.npy")	# data of sales item count
		#self.price_tensor  = np.load(path + "price.npy")  	# data of price
		self.real = self.real[:, :, :self.time_id+1]
		self.exist = np.load(path + "exist.npy")[:, :, :self.time_id+1]	# is training data or not
		self.exist_first = np.load(path + "exist_first.npy")[:, :, :self.time_id+1]	# is training data or not

		print "tensor.shape =", self.real.shape 
		print "exist.shape  =", self.exist.shape

	def preprocess(self):
		def transform_to_binary():
			binary = np.zeros(self.real.shape, dtype = float)
			
			for i in range(self.exist.shape[0]):
				for j in range(self.exist.shape[1]):
					for k in range(self.exist.shape[2]):
						if self.exist[i, j, k] == 1 and self.real[i, j, k] != 0: binary[i, j, k] = 1
			self.binary = binary

		def set_train_list():
			train = []
			valid = []
			test = []
			if self.train_all == True:
				for i in range(self.exist.shape[0]):
					for j in range(self.exist.shape[1]):
						for k in range(self.exist.shape[2]):
							if self.exist[i, j, k] == 1 and  k > 0 and np.sum(self.real[i, :, k-1]) != 0:
								if k < self.time_id: train.append([i, j, k, k-1])
								elif k == self.time_id:
									if self.exist_first[i, j, k] == 1: 
										test.append([i, j, k, k-1])
									else: valid.append([i, j, k, k-1])
				# train: < kth month
				# valid: kth month, exist_first = 0
				# test:  kth month, exist_first = 1
			else:
				for i in range(self.exist.shape[0]):
					for j in range(self.exist.shape[1]):
						for k in range(self.exist.shape[2]):
							if self.exist_first[i, j, k] == 1 and np.sum(self.real[i, :, k-1]) != 0:
								if k < self.time_id and k > 0: train.append([i, j, k, k-1])
								elif k == self.time_id:
									if self.exist_first[i, j, k] == 1: 
										test.append([i, j, k, k-1])
									else: valid.append([i, j, k, k-1])
				# train: < kth month, exist_first = 1
				# valid: -
				# test:  kth month, exist_first = 1
			self.train_list = np.array(train) # all exist, not in time_id
			self.valid_list = np.array(valid) # in time_id, not appears at the first time
			self.test_list  = np.array(test)  # in time_id, appears at the first time
			print "train =", self.train_list.shape
			print "valid =", self.valid_list.shape
			print "test =", self.test_list.shape

		transform_to_binary()
		set_train_list()
		
	def set_tensor(self):
  		print "\nSetting parameter..."
  		
  		# tensor_dim = device, item, last_item
  		self.v_di = tf.Variable(tf.random_normal([self.real.shape[0], self.k_di])) 
  		self.v_id = tf.Variable(tf.random_normal([self.real.shape[1], self.k_di])) 
  		self.v_il = tf.Variable(tf.random_normal([self.real.shape[1], self.k_il])) 
  		self.v_li = tf.Variable(tf.random_normal([self.real.shape[1], self.k_il])) 

  		self.global_bias = tf.Variable(tf.random_normal([1]), name = "global_bias")
  		self.bias_d = tf.Variable(tf.random_normal([self.real.shape[0], 1, 1])) # bias for device
  		self.bias_i = tf.Variable(tf.random_normal([1, self.real.shape[1], 1])) # bias for product

  		self.index_nn = tf.placeholder(tf.int32, [None, 4]) # index_list = (None, d, i, t)
		
	def build_model(self):
		print "\nBuilding Model..."
		
		regularization = tf.reduce_sum(tf.square(self.v_di)) + tf.reduce_sum(tf.square(self.v_id)) \
			+ tf.reduce_sum(tf.square(self.v_il)) + tf.reduce_sum(tf.square(self.v_li))
		regularization += tf.square(self.global_bias) \
			+ tf.reduce_sum(tf.square(self.bias_d)) + tf.reduce_sum(tf.square(self.bias_i))
		
  		v_di_batch = tf.gather(self.v_di, self.index_nn[:, 0]) # device
  		v_id_batch = tf.gather(self.v_id, self.index_nn[:, 1]) # item
  		v_il_batch = tf.gather(self.v_il, self.index_nn[:, 1]) # item
  		
  		bias_d_batch = tf.gather(self.bias_d, self.index_nn[:, 0]) # device
  		
  		real_trans = tf.transpose(self.binary, perm = [0, 2, 1]) # device * time * item
  		last_index_batch = tf.gather_nd(real_trans, tf.concat([self.index_nn[:, 0:1], self.index_nn[:, 3:4]], 1))
  		# from d * t * i gets index: d, t-1 --> last i_dim = 351, for products have been sold out last week
  		last_index_batch = tf.expand_dims(tf.cast(last_index_batch, tf.float32), 2)
 
  		a = tf.multiply(self.v_li, last_index_batch)
		x = tf.reduce_sum(tf.multiply(v_di_batch, v_id_batch), axis = 1) + \
			tf.reduce_sum(tf.multiply(a, tf.expand_dims(v_il_batch, 1)), axis = [1, 2])/\
			tf.reduce_sum(last_index_batch, axis = [1, 2])
		real = tf.cast(tf.gather_nd(self.real, self.index_nn[:, 0:3]), tf.float32)
		print "real =", real
		
		# loss and training
		self.pred = x
		self.real_output = real
		self.loss = tf.reduce_mean(tf.square(tf.subtract(real, x)))
		self.mae = tf.reduce_mean(tf.abs(tf.subtract(real, x)))
		self.regul_loss = regularization * self.lamb
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		self.train_regul_step = tf.train.AdamOptimizer(self.lr).minimize(self.regul_loss)

		if self.store_model == True or self.reuse_model == True: 
			self.saver = tf.train.Saver()

	def train(self):
		print "\nStart training..."
		print "Setting # =", self.record_name, "\n"
		self.init = tf.global_variables_initializer()
		self.sess = tf.Session() 
		self.sess.run(self.init)

		if self.reuse_model == True:
			self.saver = tf.train.import_meta_graph("model/model_" + str(self.record_name) + ".meta")
			self.saver.restore(self.sess, 'model/model_' + self.record_name + '')
		 	print("Model restored.")

		self.epoch = 0
		while self.epoch < self.total_epoch:
			if self.epoch % 20 == 0: self.output_loss()

			np.random.shuffle(self.train_list)
			for batch in range(int(len(self.train_list)/self.bs)):
				self.sess.run(self.train_step,
					feed_dict = {self.index_nn: self.train_list[batch*self.bs:(batch+1)*self.bs]})
			self.sess.run(self.train_regul_step)
			self.epoch += 1

			if self.store_model == True and self.epoch % 10000 == 0 and self.epoch != 0: 
				save_path = self.saver.save(self.sess, 
					'model/model_' + str(self.record_name) + '')
				print("Model saved in file: %s" % save_path)
			
		self.output_loss()
		self.output_pred()
		if self.store_model == True:
			save_path = self.saver.save(self.sess, 'model/model_' + self.record_name + '')  	
			print("Model saved in file: %s" % save_path)
		print "\nSetting # =", self.record_name
		print "Finished!\n"

	def output_loss(self):
		def output_long_list(arr, name, n):
			mse, mae = [], []
			if self.train_all == True: bs = 1000
			else:
				bs = 500
				n = 1
			for batch in range(int(len(arr)/bs/n)):
				s, a = self.sess.run([self.loss, self.mae],
					feed_dict = {self.index_nn: arr[batch*bs:(batch+1)*bs]})
				mse.append(s)
				mae.append(a)
				
			mse = np.mean(np.array(mse))
			mae = np.mean(np.array(mae))
			print "%s | mse = %.3f | mae = %.3f" %(name, mse, mae)
			self.fout.write("%s | mse = %.3f | mae = %.3f\n" %(name, mse, mae)) 

		print "epoch = %d" %(self.epoch)
		self.fout.write("epoch = %d\n" %(self.epoch))
		output_long_list(self.train_list, "Train", 10) # training
		if self.train_all == True: output_long_list(self.valid_list, "Valid", 1)  # valid
		
		# testing
		for batch in range(1):
			mse, mae = self.sess.run([self.loss, self.mae], 
				feed_dict = {self.index_nn: self.test_list})
			
		print "Test  | mse = %.3f | mae = %.3f" %(mse, mae)
		self.fout.write("Test  | mse = %.3f | mae = %.3f\n" %(mse, mae)) 

		if mae < self.min_MAE:
			self.min_epoch = self.epoch
			self.min_MAE = mae
			self.min_MSE = mse
			print "NEW: Epoch = %d, MAE = %.2f, MSE = %.2f" %(self.epoch, self.min_MAE, self.min_MSE)
			self.fout.write("NEW: Epoch = %d, MAE = %.2f, MSE = %.2f\n" %(self.epoch, self.min_MAE, self.min_MSE))
		self.fout.write("\n")

	def output_pred(self):
		self.fout.write("pred, real, abs_error, squ_error \n") 
		pred = self.sess.run(self.pred,
			feed_dict = {self.index_nn: self.test_list})
		real = self.sess.run(self.real_output,
			feed_dict = {self.index_nn: self.test_list})
		
		for i in range(pred.shape[0]):
			self.fout.write("%.3f, %.3f, " %(pred[i], real[i])) 
			self.fout.write("%.3f, " % abs(pred[i] - real[i]))
			self.fout.write("%.3f\n" % (pred[i] - real[i])**2) 

	def return_result(self):
		return self.min_epoch, self.min_MAE, self.min_MSE

def main():
	'''fout = open('record_(value)_0211', 'a')
	fout.write("\nk8, l01\n")
	for time_id in [54]:
		fpmc = FPMC(target = "amount", record_name = "k8_l01_(value)_" + str(time_id), interval = "week",
			reuse_model = False, store_model = False, total_epoch = 3000, time_id = time_id, lr = 0.003, 
			core_size = 8, lamb = 0.01)
		epoch, MAE, MSE = fpmc.return_result()
		fout.write("week = %d, epoch = %d, MAE = %.3f, MSE = %.3f\n" %(time_id, epoch, MAE, MSE))
		
	fout.write("k16, l01\n")
	for time_id in [54]:
		fpmc = FPMC(target = "amount", record_name = "test_k16_l01_(value)_" + str(time_id), interval = "week",
			reuse_model = False, store_model = False, total_epoch = 5000, time_id = time_id, lr = 0.003,
			core_size = 16, lamb = 0.01)
		epoch, MAE, MSE = fpmc.return_result()
		fout.write("week = %d, epoch = %d, MAE = %.3f, MSE = %.3f\n" %(time_id, epoch, MAE, MSE))
	
	fout.write("\nk16, l03\n")
	for time_id in [54]:
		fpmc = FPMC(target = "amount", record_name = "k16_l03_(value)_" + str(time_id), interval = "week",
			reuse_model = False, store_model = False, total_epoch = 5000, time_id = time_id, lr = 0.003, 
			core_size = 16, lamb = 0.03)
		epoch, MAE, MSE = fpmc.return_result()
		fout.write("week = %d, epoch = %d, MAE = %.3f, MSE = %.3f\n" %(time_id, epoch, MAE, MSE))
	'''
	fout = open('record_(value)_0211', 'a')
	fout.write("\nk8, l01, all\n")
	fout.close()
	for time_id in range(95, 103, 1):
		FPMC(target = "amount", record_name = "k8_l01_all_(value)_" + str(time_id), interval = "week",
			reuse_model = False, store_model = False, total_epoch = 1500, time_id = time_id, lr = 0.001, train_all = True,
			core_size = 8)

if __name__ == '__main__':
	main()


		
