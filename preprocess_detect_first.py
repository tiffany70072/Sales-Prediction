# 11/11/2017
# detect if the data appears at the first time

import tensorflow as tf
import numpy as np 
import pandas as pd
import random

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config = config))

class Model(object):
	def __init__(self):
		self.load("data-112017/")

	def load(self, path):
		print "Loading"
		self.tensor = np.load(path + "amount.npy")
		#self.mask = np.load(path + "mask.npy")
		self.exist = np.load(path + "exist.npy")
		
	def check_number(self): # number for the same device and same product in the past
		self.number = np.zeros(self.tensor.shape, dtype = int)
		print self.tensor[0, 1]
		print self.exist[0, 1]
		print np.sum(self.exist[0, 1, :13])
		
		for i in range(self.tensor.shape[0]):
			for j in range(self.tensor.shape[1]):
				for k in range(self.tensor.shape[2]):
					if self.exist[i][j][k] == 1:
						self.number[i][j][k] = np.sum(self.exist[i, j, :k])
			if i % 100 == 0: print i, "/", self.tensor.shape[0]
		print self.number[0][1][13]

	def check_first(self): 
		self.exist_before = np.zeros(self.tensor.shape, dtype = int)	 	# 0 for first, other = 1
		self.first_time = np.zeros(self.tensor.shape, dtype = int) 			# 1 for first, other = 0
		self.first_zero = np.zeros(self.tensor.shape, dtype = int) 			# 1 for amount = zero, other = 0
		for i in range(self.tensor.shape[0]):
			for j in range(self.tensor.shape[1]):
				for k in range(self.tensor.shape[2]):
					if self.exist[i][j][k] == 1 and self.number[i][j][k] != 0:
						self.exist_before[i][j][k] = 1
					if self.exist[i][j][k] == 1 and self.number[i][j][k] == 0:
						self.first_time[i][j][k] = 1
						if self.tensor[i][j][k] == 0.0:
							self.first_zero[i][j][k] = 1
			if i % 100 == 0: print i, "/", self.tensor.shape[0]
		print "# (Total data) =", np.sum(self.exist)
		print "# (Exist before) =", np.sum(self.exist_before)
		print "# (Exist the first time) =", np.sum(self.exist) - np.sum(self.exist_before)
		for i in range(24):
			print i, np.sum(self.exist[:, :, i]) - np.sum(self.exist_before[:, :, i])

		#np.save("data-112017/exist_before", self.exist_before)
		np.save("data-112017/exist_first", self.first_time)
		np.save("data-112017/first_zero", self.first_zero)
		exit()

	def check_amount_zero(self):
		# count how many zero are in the first data
		self.count_amount_zero = np.zeros(self.tensor.shape[2], dtype = int)
		for k in range(self.tensor.shape[2]):
			for i in range(self.tensor.shape[0]):
				for j in range(self.tensor.shape[1]):
					if self.exist[i][j][k] == 1 and self.exist_before[i][j][k] == 0 and self.tensor[i][j][k] == 0.0:
						self.count_amount_zero[k] += 1
			print "Month =", k, ", #(Amount = 0) = %4d" %self.count_amount_zero[k], 
			num_first = np.sum(self.exist[:, :, k]) - np.sum(self.exist_before[:, :, k])
			print ", #(First time) = %5d" %num_first,
			print ", %.3f" %(self.count_amount_zero[k]/float(num_first))

model = Model()

model.check_number()
model.check_first()
model.check_amount_zero()
#model.check_each()
#model.check_price() # check if price is zero
#model.check_amount()
	
