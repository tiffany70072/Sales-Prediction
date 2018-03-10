# Original file name: classifier_to_value_0226
# Method: New matrix factorization and factorization machine
# Use the features as same as classifier's
# Training data: the real first time

import tensorflow as tf
import numpy as np 
import random
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate
from keras.regularizers import l2
from keras import optimizers

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config = config))
random.seed(5)

class Prediction_Model(object):
	def __init__(self, interval = "week", test_id = 50, previous_num = 4, 
				total_epoch = 30, record_name = None, derivative = False,
				mf = None, fm = None):

		# constant
		self.lr = 0.001
		self.previous_num = previous_num			# how many weeks we need for previous "amount"
				
		self.interval = interval
		self.derivative = derivative
		self.mf = mf
		self.fm = fm
		if self.interval == "month": self.tensor_shape = [499, 433, 24]
		elif self.interval == "week": self.tensor_shape = [499, 433, 103]
		
		self.input_dim_onehot = 499 + 433				
		self.input_dim_value = self.previous_num * 2 + 5 
		if self.derivative == True: self.input_dim_value += (self.previous_num - 1) * 2
		
		#self.input_dim = self.input_dim_value + self.input_dim_one
		self.record_name  = record_name
		self.total_epoch  = total_epoch
		
		# function
		self.load()

		if self.fm == True: # factorization machine
			self.preprocess_fm() 
			
		if self.mf == True: # matrix factorization
			self.preprocess_mf() 
			
		for self.test_id in range(test_id, 103, 1): 	# which week we are going to predict, from 0 ~ 103
			if self.mf == True:
				self.set_data_mf()
				self.build_graph()
				self.train_mf()
			
			if self.fm == True:
				self.set_data()
				self.set_data_fm()
				if self.mf == True: 
					self.combination()
					self.train_fm(use_mf = True)
				else: self.train_fm()

	def load(self):
		print "\nLoading"
		if self.interval == "week": 
			self.amount = np.load("data-122017-week/amount.npy")
			self.exist  = np.load("data-122017-week/exist.npy")
			self.exist_first = np.load("data-122017-week/exist_first.npy") 	# 1: data exist first time, 0: others
		
		print "raw_data =", self.amount.shape

	def preprocess_fm(self):
		'''
		1. device_id (one hot)
		2. product_id (one hot)
		3. same device average amount for 4 weeks (4 values)
		4. same product average amount for 4 weeks (4 values)
		5. same device data num (1 value)
		6. same product data num (1 value)
		7. same device/ product first time sales amount (various dim?) --> do average?  (2 values)
		8. same product first time num (1 value) 
		9. same device/ product 1st derivative previous amount
		'''
		self.n = np.sum(self.exist_first)
		self.data_index = np.zeros([self.n, 2], dtype = int)
		self.data_onehot = np.zeros([self.n, self.input_dim_onehot], dtype = float) 
		self.data = np.zeros([self.n, self.input_dim_value], dtype = float)
		self.y = np.empty([self.n], dtype = float)
		self.time_id_list = np.zeros([self.n], dtype = float)
		count = 0

		for k in range(self.previous_num, self.exist.shape[2]):
			for i in range(self.exist.shape[0]):
				for j in range(self.exist.shape[1]):
					if self.exist_first[i, j, k] == 1:
						self.data_index[count][0] = i
						self.data_index[count][1] = j
						self.data_onehot[count, i] = 1
						self.data_onehot[count, 499 + j] = 1

						for k0 in range(self.previous_num): # 3
							if np.sum(self.exist[i, :, k-k0-1]) != 0:
								self.data[count][k0] = np.sum(np.multiply(self.amount[i, :, k-k0-1], self.exist[i, :, k-k0-1]))/np.sum(self.exist[i, :, k-k0-1])
						for k0 in range(self.previous_num): # 4
							if np.sum(self.exist[:, j, k-k0-1]) != 0:
								self.data[count][self.previous_num + k0] = np.sum(np.multiply(self.amount[:, j, k-k0-1], self.exist[:, j, k-k0-1]))/np.sum(self.exist[:, j, k-k0-1])
						self.data[count][self.previous_num * 2 + 0] = np.sum(self.exist[i, :, k-4:k])
						self.data[count][self.previous_num * 2 + 1] = np.sum(self.exist[:, j, k-4:k])
						if np.sum(self.exist_first[i, :, :k]) != 0: # 7
							self.data[count][self.previous_num * 2 + 2] = np.sum(np.multiply(self.amount[i, :, :k], self.exist_first[i, :, :k]))/np.sum(self.exist_first[i, :, :k])
						if np.sum(self.exist_first[:, j, :k]) != 0: # 7
							self.data[count][self.previous_num * 2 + 3] = np.sum(np.multiply(self.amount[:, j, :k], self.exist_first[:, j, :k]))/np.sum(self.exist_first[:, j, :k])
						self.data[count][self.previous_num * 2 + 4] = np.sum(self.exist_first[:, j, :k])
						
						if self.derivative == True: 
							for k0 in range(self.previous_num - 1): # 9
								self.data[count][self.previous_num * 2 + 5 + k0] = self.data[count][k0+1] - self.data[count][k0]
								self.data[count][self.previous_num * 3 + 4 + k0] = self.data[count][self.previous_num + k0+1] - self.data[count][self.previous_num + k0]

						self.y[count] = self.amount[i, j, k]
						self.time_id_list[count] = k
						count += 1
		print "data =", self.data.shape
		print self.data[:10]
		print "data_onehot =", self.data_onehot.shape
		print "y =", self.y.shape, self.y[:100]
		print "time_id_list =", self.time_id_list[:10]
		np.save("data-022018-value/value_prediction_data", self.data)
		np.save("data-022018-value/value_prediction_onehot", self.data_onehot)
		np.save("data-022018-value/value_prediction_y", self.y)
		np.save("data-022018-value/value_prediction_time_list", self.time_id_list)

	def set_data(self):
		flag = 0
		for i in range(self.n): 
			if flag == 0 and self.time_id_list[i] == self.test_id:
				self.train_num = i
				flag = 1
			if flag == 1 and self.time_id_list[i] == self.test_id + 1:
				self.test_num = i
				break
		print "test_id =", self.test_id
		print "train_num =", self.train_num
		print "test_num =", self.test_num
		self.test_num = self.test_num - self.train_num

	def set_data_fm(self):
		self.x_train_onehot = self.data_onehot[:self.train_num]
		self.x_test_onehot = self.data_onehot[self.train_num:self.train_num+self.test_num]
		self.x_train = self.data[:self.train_num]
		self.x_test = self.data[self.train_num:self.train_num+self.test_num]
		self.y_train = self.y[:self.train_num]
		self.y_test = self.y[self.train_num:self.train_num+self.test_num]
		self.x_train_index = self.data_index[:self.train_num]
		self.x_test_index = self.data_index[self.train_num:self.train_num+self.test_num]
		
	def preprocess_mf(self):
		self.matrix_first_real  = np.zeros([self.exist.shape[0], self.exist.shape[1]], dtype = float) # real value
		self.matrix_first_exist = np.zeros([self.exist.shape[0], self.exist.shape[1]], dtype = float) # whether exist
		self.matrix_first_time  = np.zeros([self.exist.shape[0], self.exist.shape[1]], dtype = float) # which weeks
		for i in range(self.exist.shape[0]):
			for j in range(self.exist.shape[1]):
				for k in range(self.exist.shape[2]):
					if self.exist_first[i][j][k] == 1:
						self.matrix_first_exist[i][j] = 1
						self.matrix_first_time[i][j]  = k
						self.matrix_first_real[i][j]  = self.amount[i][j][k]
		print np.sum(self.matrix_first_exist), np.sum(self.exist_first)

	def set_data_mf(self):
		self.mask_train = np.zeros([self.exist.shape[0], self.exist.shape[1]], dtype = float)
		self.mask_test  = np.zeros([self.exist.shape[0], self.exist.shape[1]], dtype = float)
		for i in range(self.exist.shape[0]):
			for j in range(self.exist.shape[1]):
				if self.matrix_first_exist[i][j] == 1 and self.matrix_first_time[i][j] < self.test_id:
					self.mask_train[i][j] = 1
				elif self.matrix_first_exist[i][j] == 1 and self.matrix_first_time[i][j] == self.test_id:
					self.mask_test[i][j] = 1
		print np.sum(self.mask_train)
		print np.sum(self.mask_test)
		print self.test_id
		#exit()

	def build_graph(self):
		self.latent_dim = 16
		self.lamb1 = 0.003
		self.lamb3 = 0.1

		self.matrix1 = tf.Variable(tf.random_normal([self.exist.shape[0], self.latent_dim])) # device property
  		self.matrix2 = tf.Variable(tf.random_normal([self.exist.shape[1], self.latent_dim])) # product property
  		self.global_bias =  tf.Variable(tf.random_normal([1]), name = "global_bias")
  		self.bias1 = tf.Variable(tf.random_normal([self.exist.shape[0], 1])) # bias for device
  		self.bias2 = tf.Variable(tf.random_normal([1, self.exist.shape[1]])) # bias for product
  		self.mask_nn = tf.placeholder(tf.float32, self.matrix_first_exist.shape)

  		self.pred = tf.einsum('im,jm->ij', self.matrix1, self.matrix2)
		regularization = tf.reduce_sum(tf.square(self.matrix1)) + tf.reduce_sum(tf.square(self.matrix2))
		
		# add biases
		self.pred = self.pred + self.global_bias + self.bias1 + self.bias2 
		regularization += tf.square(self.global_bias) \
				+ tf.reduce_sum(tf.square(self.bias1)) + tf.reduce_sum(tf.square(self.bias2))
		regularization *= self.lamb1
		non_nagative = self.lamb3 * tf.reduce_sum(self.mask_nn*(tf.abs(self.pred) - self.pred))
		
		# loss and error (for output)
		matrix_first_real = tf.cast(self.matrix_first_real, tf.float32)
		count = tf.reduce_sum(self.mask_nn)
		loss = tf.reduce_sum(tf.square(self.mask_nn*tf.subtract(matrix_first_real, self.pred)))/tf.cast(count, tf.float32) # real mean
		self.mse = tf.reduce_sum(tf.square(self.mask_nn * tf.subtract(matrix_first_real, self.pred)))/tf.cast(count, tf.float32)
		self.mae = tf.reduce_sum(tf.abs   (self.mask_nn * tf.subtract(matrix_first_real, self.pred)))/tf.cast(count, tf.float32)
		
		self.loss = loss + regularization + non_nagative # loss for training
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def train_mf(self):
		self.min_epoch = 0
		self.min_MAE = 100000
		self.min_MSE = 100000

		print "\nStart training matrix factorization..."
		self.init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(self.init)

		self.epoch = 0
		while self.epoch < 20000:
			if self.epoch % 500 == 0: self.output_loss()
			self.sess.run(self.train_step, feed_dict = {self.mask_nn: self.mask_train})
			self.epoch += 1
			
		self.output_loss()
		fout = open('record_valuePrediction_mf_50_102_d100', 'a')
		fout.write("Week = %d, %d\t%.3f\t%.3f\n" %(self.test_id, self.min_epoch, self.min_MAE, self.min_MSE))
		fout.close()
		#self.output_pred()
		print "Finished!\n"

	def output_loss(self):
		# output loss
		loss = self.sess.run(self.loss, feed_dict = {self.mask_nn: self.mask_train})
		print "Epoch = %d | loss = %.3f" %(self.epoch, loss)
		#self.fout.write("Epoch = %d | loss = %.3f\n" %(str(self.epoch), loss)) 

		# output train error
		mse, mae = self.sess.run([self.mse, self.mae], feed_dict = {self.mask_nn: self.mask_train})	 
		print "Train | MSE = %.3f, MAE = %.3f" %(mse, mae)
		#self.fout.write("Train | MSE = %.3f, MAE = %.3f\n" % (mse, mae))

		# output test error
		#pred = self.sess.run(self.pred, feed_dict = {self.real_nn: self.real_2d})
		mse, mae = self.sess.run([self.mse, self.mae], feed_dict = {self.mask_nn: self.mask_test})	 
		print "Test  | MSE = %.3f, MAE = %.3f" %(mse, mae)
		#self.fout.write("Test  | MSE = %.3f, MAE = %.3f\n" % (mse, mae))

		if mae < self.min_MAE:
			self.min_epoch = self.epoch
			self.min_MAE = mae
			self.min_MSE = mse

			pred_list = self.sess.run(self.pred)
			self.min_ans = pred_list
			#for [i, j, k] in self.test_list: self.min_ans.append(pred[i][j][k])
			print "NEW: Epoch = %d, MAE = %.2f, MSE = %.2f" %(self.epoch, self.min_MAE, self.min_MSE)
			
	def output_pred(self):
		pred_list = self.sess.run(self.pred)
		print pred_list.shape
		for i in range(self.exist.shape[0]):
			for j in range(self.exist.shape[1]):
				if self.mask_test[i][j] == 1:
					real = self.matrix_first_real[i][j]
					pred = pred_list[i][j]
					#print 333
					#print real
					#print pred
					print "%.3f, %.3f, %.3f %.3f" %(real, pred, abs(real - pred), (real - pred) ** 2)

	'''def normalization(self):
		mean = np.mean(self.value_train, axis = 0)
		std = np.std(self.value_train, axis = 0)
		self.value_train = (self.value_train - mean[:self.dim_value])/std[:self.dim_value]
		self.value_test = (self.value_test - mean[:self.dim_value])/std[:self.dim_value]
		self.x_train[:, :self.dim_value] = (self.x_train[:, :self.dim_value] - mean[:self.dim_value])/std[:self.dim_value]
		self.x_test[:, :self.dim_value] = (self.x_test[:, :self.dim_value] - mean[:self.dim_value])/std[:self.dim_value]
	'''
	def shuffle_in_unison(self, a, b):
	    assert len(a) == len(b)
	    shuffled_a = np.empty(a.shape, dtype = a.dtype)
	    shuffled_b = np.empty(b.shape, dtype = b.dtype)
	    permutation = np.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	def train_fm(self, use_mf = False):
		# model 1 for using input data with values
		inputs_value = Input(shape = (self.input_dim_value,))
		x1 = Dense(100, activation = 'relu', kernel_regularizer = l2(0.0))(inputs_value)
		#x1 = Dropout(0.2)(x1)
		x1 = Dense(100, activation = 'relu', kernel_regularizer = l2(0.0))(x1)
		x1 = Dropout(0.2)(x1)

		# model 2 for using one hot input data
		inputs_one = Input(shape = (self.input_dim_onehot,))
		x2 = Dense(100, kernel_regularizer = l2(0.0), activation = 'relu')(inputs_one)
		
		# concatenate two models
		x = concatenate([x1, x2])
		#x = Dense(100, W_regularizer = l2(0.00), activation = 'relu')(x)
		prediction = Dense(1, kernel_regularizer = l2(0.0))(x)
		if use_mf == True:
			inputs_mf = Input(shape = (1,))
			x = concatenate([inputs_mf, prediction])
			x = Dense(10)(x)
			prediction = Dense(1)(x) 
		
		if use_mf == False:
			model = Model(inputs = [inputs_value, inputs_one], outputs = prediction)
	
			model.summary()
			print "\nStart training"
			opt = optimizers.Adam(lr = 3E-4)
			model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['mae', 'mse'])
			model.fit([self.x_train, self.x_train_onehot], self.y_train, batch_size = 128, epochs = 20, shuffle = True, 
				validation_data = ([self.x_test, self.x_test_onehot], self.y_test))

			self.pred_fm = model.predict([self.x_train, self.x_train_onehot])[:1000]
			self.output_result(self.y_train[:1000], "train")

			self.pred_fm = model.predict([self.x_test, self.x_test_onehot])
			self.output_result(self.y_test, "test")

		else:
			model = Model(inputs = [inputs_value, inputs_one, inputs_mf], outputs = prediction)
	
			model.summary()
			print "\nStart training"
			opt = optimizers.Adam(lr = 3E-4)
			print self.x_train_mf.shape, self.x_test_mf.shape
			model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['mae', 'mse'])
			model.fit([self.x_train, self.x_train_onehot, self.x_train_mf], self.y_train, batch_size = 128, epochs = 20, shuffle = True, 
				validation_data = ([self.x_test, self.x_test_onehot, self.x_test_mf], self.y_test))

			self.pred_fm = model.predict([self.x_train, self.x_train_onehot, self.x_train_mf])[:1000]
			self.output_result(self.y_train[:1000], "train")

			self.pred_fm = model.predict([self.x_test, self.x_test_onehot, self.x_test_mf])
			self.output_result(self.y_test, "test")

	def output_result(self, real_list, name):
		MAE = 0
		MSE = 0
		print "pred =", self.pred_fm.shape
		print "real =", real_list.shape
		for i in range(self.pred_fm.shape[0]):
			real = real_list[i]
			pred = self.pred_fm[i][0]
			MAE += abs(real - pred)
			MSE += (real - pred) ** 2
			if i < 30: print "%.3f, %.3f, %.3f %.3f" %(real, pred, abs(real - pred), (real - pred) ** 2)
		MAE /= float(self.pred_fm.shape[0])
		MSE /= float(self.pred_fm.shape[0])
			
		print "MAE = %.3f, " %MAE,
		print "MSE = %.3f" %MSE

		if name == "test":
			fout = open('record_mf_fm_50_102_d100', 'a')
			fout.write("Week = %d\t" %self.test_id)
			fout.write("%.3f\t%.3f\n" %(MAE, MSE))
			fout.close()

		'''if name == "test" and self.mf == True:
			fout = open('record_mf_fm_d100_' + str(self.test_id), 'a')
			fout.write("FM: Week = %d\n" %(self.test_id))
			fout.write("pred, real, mae, mse\n")
			for i in range(self.pred_fm.shape[0]):
				real = real_list[i]
				pred = self.pred_fm[i][0]
				MAE += abs(real - pred)
				MSE += (real - pred) ** 2
				#if i < 30: print "%.3f, %.5f, %.3f %.3f" %(real, pred, abs(real - pred), (real - pred) ** 2)
				fout.write("%.3f\t%.3f\t%3f\t%3f\n" %(pred, real, abs(pred - real), (pred - real)**2))
			fout.close()'''

	def combination(self):
		'''fout = open('record_mf_fm_d100_' + str(self.test_id), 'a')
		fout.write("MF: Week = %d, %d\t%.3f\t%.3f\n" %(self.test_id, self.min_epoch, self.min_MAE, self.min_MSE))
		fout.write("pred, real, mae, mse\n")
		for i in range(self.exist.shape[0]):
			for j in range(self.exist.shape[1]):
				if self.mask_test[i][j] == 1:
					pred = self.min_ans[i][j]
					real = self.matrix_first_real[i][j]
					fout.write("%.3f\t%.3f\t%3f\t%3f\n" %(pred, real, abs(pred - real), (pred - real)**2))	
		fout.close()'''

		self.x_train_mf = np.empty([self.x_train_index.shape[0], 1], dtype = float)
		self.x_test_mf = np.empty([self.x_test_index.shape[0], 1], dtype = float)

		for i in range(self.x_train_index.shape[0]):
			self.x_train_mf[i][0] = self.min_ans[self.x_train_index[i][0]][self.x_train_index[i][1]]

		for i in range(self.x_test_index.shape[0]):
			self.x_test_mf[i][0] = self.min_ans[self.x_test_index[i][0]][self.x_test_index[i][1]]

		
def main():
	Prediction_Model(interval = "week", test_id = 50, fm = True, mf = True)
	exit()
	
	Prediction_Model(interval = "week", test_id = 95, derivative = False)
	fout = open('record_valuePrediction_50_102', 'a')
	fout.write("Previous_num = 8\n")
	fout.close()
	Prediction_Model(interval = "week", previous_num = 8, test_id = 50, derivative = False)
	fout = open('record_valuePrediction_50_102', 'a')
	fout.write("Previous_num = 4, Derivative = True\n")
	fout.close()
	Prediction_Model(interval = "week", previous_num = 4, test_id = 50, derivative = True)

	fout = open('record_valuePrediction_50_102', 'a')
	fout.write("Previous_num = 4, Derivative = True, epoch = 30\n")
	fout.close()
	Prediction_Model(interval = "week", previous_num = 4, test_id = 50, derivative = True, total_epoch = 30)

if __name__ == '__main__':
	main()

	
		