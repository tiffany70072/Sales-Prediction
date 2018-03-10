# Method: binary classifier
# Original file name: classifier_1218
# Predict the sales amount is zero or not
# Training data: all data

import tensorflow as tf
import numpy as np 
import random
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate
from keras.regularizers import l2
from keras import optimizers

from sklearn.svm import SVC

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config = config))
random.seed(5)


class Classifier_Model(object):
	def __init__(self, previous_time = 1, testing_time_id = 23,
				total_epoch = 30, model = "nn", interval = "month",
				add_data = False, use_all_data = False, 
				add_date_feature = False, loss = 'categorical_crossentropy',
				record_name = None, combination = False):

		# constant
		self.lr = 0.001
		self.previous_num = previous_time			# how many month we need for previous "amount"
		self.testing_time_id = testing_time_id 		# which month we are going to predict, from 0 ~ 23
		self.interval = interval
		if self.interval == "month": self.tensor_shape = [499, 433, 24]
		elif self.interval == "week": self.tensor_shape = [499, 433, 103]
		
		#self.dim_value = self.previous_num + 1 								# previous amount and current price
		#self.input_dim_one 	 = self.tensor_shape[0] + self.tensor_shape[1] 	# device and product id			
		#self.input_dim = self.input_dim_value + self.input_dim_one
		#self.file = str(self.previous_num) + "_" + str(self.testing_time_id)
		#self.name = "data-112017/classifier_zero_" + self.file

		self.use_all_data = use_all_data 	# True: use all data, False: only use the first time data
		self.add_data 	  = add_data		# balance training data of each class
		self.record_name  = record_name
		self.total_epoch  = total_epoch
		self.model 		  = model 			# 'nn' or 'svm'
		self.add_date_feature = add_date_feature 			# add feature of which month
		self.loss 		  = loss
		self.combination  = combination
		self.interval = interval
		
		# function
		self.load()
		self.set_normal()
		self.set_data()
		self.set_testing_data()
		
		if self.y_test.shape[0] != 0: 
			#self.output_data(30, self.x_test)
			self.normalization()
			if model == 'nn': self.train_nn()
			elif model == 'svm': self.train_svm()

	def load(self):
		print "\nLoading"
		if self.interval == "month": 
			self.data = np.load("data-112017/classifier_23_2.npy")
			self.exist_first = np.load("data-112017/exist_first.npy") 	# 1: data exist first time, 0: others
		elif self.interval == "week": 
			if self.use_all_data == False: self.data = np.load("data-122017-week/classifier_week.npy")
			else: self.data = np.load("data-122017-week/classifier_week_all.npy")
			self.exist_first = np.load("data-122017-week/exist_first.npy") 	# 1: data exist first time, 0: others
		
		remove = [17, 18, 21, 22, 19, 20, 23, 24, 6, 8, 10, 12] 
		self.data = np.delete(self.data, remove, 1)
		print "data =", self.data.shape, ", remove =", remove

	def output_data(self, n, data = None):
		print "real, price,", 
		print "amount(previous 1), amount(previous), amount(previous first),",
		print "times(previous), times(previous first),",
		print "amount(device previous first), times(device previous), times(device previous first),"
		print "ratio(!=zero, product), ratio(!=zero, device), "
		for i in range(n): 
			print self.y_test[i], "\t", 
			for j in range(0, self.dim_value): print "%.1f\t" %int(data[i][j]),
			print ""

	def set_test_list(self):
		self.test_list = []
		for i in range(self.mask.shape[0]):
			for j in range(self.mask.shape[1]):
				if self.first[i][j][self.testing_time_id] == 1:
					self.test_list.append([i, j, self.testing_time_id])
		print "#(Testing) =", len(self.test_list)

	def set_mask(self):
		# remove data of the testing month from training data
		for i in range(self.mask.shape[0]):
			for j in range(self.mask.shape[1]):
				for k in range(self.testing_time_id, self.mask.shape[2], 1):
					if self.mask[i, j, k] == 1: self.mask[i, j, k] = 0

		# remove previous data of testing data from training data
		if self.use_all_data == True:
			for i in range(len(self.test_list)):
				for k in range(0, self.testing_time_id, 1):
					if self.mask[self.test_list[i][0], self.test_list[i][1], k] == 1: 
						self.mask[self.test_list[i][0], self.test_list[i][1], k] = 0

	def set_normal(self):
		#const1 = 1/((2.0*math.pi)**2*sigma)*160
		if self.interval == "month": 
			self.n = 2
			sigma = 1
		elif self.interval == "week":
			self.n = 8
			sigma = 3

		const1 = 1.0
		const2 = -1/2.0/(sigma**2)
		print const1, const2
		self.normal = []
		week_list = []
		for i in range(-self.n, self.n+1, 1): week_list.append(i)
		for week_id in week_list:
			print week_id, "%.3f" %(const1*math.exp(week_id**2*const2))
			self.normal.append(const1*math.exp(week_id**2*const2))

	def set_data(self):
		print "\nSet data..."
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []
		self.test_list = []
		self.dim_onehot = 499 + 433
		if self.add_date_feature == True and self.interval == "month":
			self.dim_onehot += 12
		if self.add_date_feature == True and self.interval == "week":
			self.dim_onehot += 52

		self.dim_value = self.data.shape[1] - 4
		print "dim_value =", self.dim_value

		zero = []
		for i in range(self.dim_onehot + self.dim_value): zero.append(0.0)
					
		for i in range(self.data.shape[0]):
			temp = np.copy(zero)
			temp[:self.dim_value] = self.data[i][-self.dim_value:]
			#print temp[:self.dim_value]
			#print self.data[i]
			temp[self.dim_value + int(self.data[i][0])] = 1
			temp[self.dim_value + 499 + int(self.data[i][1])] = 1
			if self.add_date_feature == True and self.interval == "month":
				for j in range(-self.n, self.n+1, 1):
					temp[self.dim_value + 499 + 433 + int(self.data[i][2]+j)%12] = self.normal[j+self.n]
				
			if self.add_date_feature == True and self.interval == "week":
				for j in range(-self.n, self.n+1, 1):
					#print int(self.data[i][2]+j), int(self.data[i][2]+j)%52, j+self.n, temp.shape, len(self.normal), self.dim_value, self.dim_onehot
					temp[self.dim_value + 499 + 433 + int(self.data[i][2]+j)%52] = self.normal[j+self.n]

			if self.data[i][3] == 1: temp_y = [0, 1] # amount != 0, index = 1
			else:					 temp_y = [1, 0] # amount = 0, index = 0

			if self.data[i][2] < self.testing_time_id:
				self.x_train.append(temp)
				self.y_train.append(temp_y)
			elif self.data[i][2] == self.testing_time_id and\
				self.exist_first[int(self.data[i][0])][int(self.data[i][1])][int(self.data[i][2])] == 1:
				self.x_test.append(temp)
				self.y_test.append(temp_y)
				self.test_list.append([self.data[i][0], self.data[i][1], self.data[i][2]])

			if i % 1000 == 0: print i,
		
		print "#(Train, before adding) =", len(self.x_train), len(self.y_train)
	
		# add (repeat) data for balancing number of zero data
		if self.add_data == True:
			num_training = np.sum(np.array(self.y_train))
			num_zero = np.sum(np.array(self.y_train)[:, 1]) # real value = zero
			num_adding = (num_training - num_zero)/num_zero - 1
			print "#(Adding zero data) =", num_adding
			for i in range(len(self.x_train)):
				if self.y_train[i][1] == 1:
					for j in range(num_adding):
						self.x_train.append(self.x_train[i])
						self.y_train.append(self.y_train[i])
		
		self.x_train = np.array(self.x_train)
		self.y_train = np.array(self.y_train)
		for i in range(2): self.x_train, self.y_train = self.shuffle_in_unison(self.x_train, self.y_train)
		
		print "x_train.shape =", self.x_train.shape
		print "y_train.shape =", self.y_train.shape
		print "#(!=zero in training data) =", np.sum(self.y_train[:, 0])
		print "#(Zero in training data) =", np.sum(self.y_train[:, 1]) # [not zero, zero]
		
		self.value_train = self.x_train[:, :self.dim_value]
		self.one_train = self.x_train[:, self.dim_value:]
		print "value and one", self.value_train[0], self.one_train[0, :10]
		
	def set_testing_data(self):
		print "\nSet Testing Data..."
	
		self.x_test = np.array(self.x_test)
		self.y_test = np.array(self.y_test)
		print "x_test.shape =", self.x_test.shape
		print "y_test.shape =", self.y_test.shape

		self.value_test = self.x_test[:, :self.dim_value]
		self.one_test = self.x_test[:, self.dim_value:]
		
		#np.save(self.name + "_x_test", self.x_test)
		#np.save(self.name + "_y_test", self.y_test)

	def normalization(self):
		mean = np.mean(self.value_train, axis = 0)
		std = np.std(self.value_train, axis = 0)
		self.value_train = (self.value_train - mean[:self.dim_value])/std[:self.dim_value]
		self.value_test = (self.value_test - mean[:self.dim_value])/std[:self.dim_value]
		self.x_train[:, :self.dim_value] = (self.x_train[:, :self.dim_value] - mean[:self.dim_value])/std[:self.dim_value]
		self.x_test[:, :self.dim_value] = (self.x_test[:, :self.dim_value] - mean[:self.dim_value])/std[:self.dim_value]

	def shuffle_in_unison(self, a, b):
	    assert len(a) == len(b)
	    shuffled_a = np.empty(a.shape, dtype = a.dtype)
	    shuffled_b = np.empty(b.shape, dtype = b.dtype)
	    permutation = np.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	def train_nn(self):
		acti_func = 'relu'

		# model 1 for using input data with values
		inputs_value = Input(shape = (self.dim_value,))
		x1 = Dense(100, activation = 'relu', kernel_regularizer = l2(0.1))(inputs_value)
		x1 = Dropout(0.5)(x1)
		#x1 = Dense(100, activation = 'relu', kernel_regularizer = l2(0.1))(x1)
		#x1 = Dropout(0.5)(x1)

		# model 2 for using one hot input data
		inputs_one = Input(shape = (self.dim_onehot,))
		x2 = Dense(200, W_regularizer = l2(0.0), activation = 'relu')(inputs_one)
		#x2 = Dropout(0.2)(x2)
		#x2 = Dense(100, W_regularizer = l2(0.0), activation = acti_func)(x2)
		#x2 = Dropout(0.2)(x2)

		# concatenate two models
		x = concatenate([x1, x2])
		x = Dense(2, W_regularizer = l2(0.1))(x)
		#x = Dropout(0.2)(x)
		prediction = Activation('softmax')(x)
			
		model = Model(inputs = [inputs_value, inputs_one], outputs = prediction)
	
		model.summary()
		print "\nStart training"
		opt = optimizers.Adam(lr = 10E-4)
		loss = self.loss
		model.compile(loss = loss, optimizer = opt, metrics = ['accuracy'])
		#model.fit([self.value_train, self.one_train], self.y_train, batch_size = 128, epochs = self.total_epoch, shuffle = True, 
		#	validation_data = ([self.value_test, self.one_test], self.y_test))
		model.fit([self.value_train, self.one_train], self.y_train, batch_size = 128, epochs = self.total_epoch, shuffle = True, 
			validation_split = 0.1)

		#self.result = model.predict([self.value_train, self.one_train])
		#self.output_result(self.y_train, "train")

		self.result = model.predict([self.value_test, self.one_test])
		self.output_result(self.y_test, "test", threshold = 0.5)
		#self.output_result(self.y_test, "test", threshold = 0.8)
		#self.output_result(self.y_test, "test", threshold = 0.9)
		#self.output_result(self.y_test, "test", threshold = 0.99)
		#self.output_result(self.y_test, "test", threshold = 0.2)

	def train_svm(self):
		clf = SVC()
		clf.fit(self.x_train, np.argmax(self.y_train, axis = 1))

		self.result = clf.predict(self.x_train) 
		self.output_result(self.y_train, 'svm_train')
		self.result = clf.predict(self.x_test) 
		self.output_result(self.y_test, 'svm_test')
		print "result =", self.result.shape
		print self.result[:30]
		print "real y_test =",  np.argmax(self.y_test [:30], axis = 1)
		print "real y_train =", np.argmax(self.y_train[:30], axis = 1)
		print "balance y_test =",  np.sum(np.argmax(self.y_test,  axis = 1))/float(self.y_test.shape[0])
		print "balance y_train =", np.sum(np.argmax(self.y_train, axis = 1))/float(self.y_train.shape[0])

	def output_result(self, real, name, threshold = 0.5):
		#self.fout = open('record_class_' + self.file, 'w')
		#self.fout.write("pred, real, abs_error, squ_error \n") 
		print "\n", name, "threshold =", threshold

		if self.model == 'nn': 
			a = np.argmax(self.result, axis = 1)
		else: a = self.result
		b = np.argmax(real, axis = 1)
		
		a = []
		for i in range(self.result.shape[0]):
			if self.result[i][0] > threshold: a.append(0)
			else: a.append(1)
			#print "%.2f, %.2f" %(self.result[i][0], self.result[i][1])
		
		#print self.result[:8]
		#print "Pred =", a[:30]
		#print "Real =", b[:30]
		#print "Are all predictions the same (percent) ?", np.sum(a)
		#print ans[:10]
		count = 0
		count1 = 0 # real (or pred) = 1, amount != 0
		count0 = 0 # real (or pred) = 0, amount = 0
		count1to1 = 0 # real = 1, pred = 1
		count0to0 = 0 # real = 0, pred = 0
		
		for i in range(self.result.shape[0]):
			if a[i] == b[i]: count += 1
			'''if b[i] == 1: 
				count1 += 1
				if a[i] == 1: count1to1 += 1
			if b[i] == 0: 
				count0 += 1
				if a[i] == 0: count0to0 += 1'''

			if a[i] == 1: 
				count1 += 1
				if b[i] == 1: count1to1 += 1
			if a[i] == 0: 
				count0 += 1
				if b[i] == 0: count0to0 += 1
			
			#self.fout.write("%.3f, %.3f, " %(self.result[i], ans[i])) 
			#self.fout.write("%.3f, " % abs(self.result[i] - ans[i]))
			#self.fout.write("%.3f\n" % (self.result[i] - ans[i])**2) 
		print "#(Correct) =", count, ", #(Total) =", self.result.shape[0],
		print ", accuracy = %.4f" %(count/float(self.result.shape[0]))
		'''print "#(Real = 1) =", count1, " (amount != 0)"
		print '#(Real = 0) =', count0, " (amount = 0)"
		print "#(Pred = 1|Real = 1) =", count1to1
		print '#(Pred = 0|Real = 0) =', count0to0
		print "Acc(Pred = 1|Real = 1) =", count1to1/float(count1)
		print "Acc(Pred = 0|Real = 0) =", count0to0/float(count0)'''

		print "#(Pred = 1) =", count1, " (amount != 0)"
		print '#(Pred = 0) =', count0, " (amount = 0)"
		print "#(Real = 1|Pred = 1) =", count1to1
		print '#(Real = 0|Pred = 0) =', count0to0
		#print "Acc(Real = 1|Pred = 1) =", count1to1/float(count1)
		#print "Acc(Real = 0|Pred = 0) =", count0to0/float(count0)
		global total_num
		global total_correct
		total_num += self.result.shape[0]
		total_correct += count
		fout = open('record_class_week_10_102_newdata' + "", 'a')
		#fout = open('record_class_month_15_23' + "", 'a')
		fout.write("Testing = %d: " %self.testing_time_id)
		fout.write("#(Correct) = %d, #(Total) = %d" %(count, self.result.shape[0]))
		fout.write(", acc = %.4f\n" %(count/float(self.result.shape[0])))
		fout.close()

		if self.combination == True: self.output_combination(a, threshold)

		#self.fout.write("Error: \n") 
		#self.fout.write("Mean Abs Err = %.3f\n" %abso)
		#self.fout.write("Mean Square Error = %.3f\n" %square)

	def output_combination(self, pred, threshold):
		# store the results of classification, for next step: combination
		# result.shape = [len(testing data), 4]
		# col name = device id, product id, 
		# 			month (default is all the same), 
		# 			is zero or not: 1 for !0, 0 for predicting zero
		print "Save the results for combination"
		print "#(Testing data) =", self.x_test.shape[0]

		result = np.empty([len(self.test_list), 4], dtype = int)
		for i in range(self.x_test.shape[0]):
			result[i][0] = self.test_list[i][0]
			result[i][1] = self.test_list[i][1]
			result[i][2] = self.test_list[i][2]
			result[i][3] = pred[i]
			#print result[i]

		if self.interval == "month":
			np.save("combination/combin_class_" + self.record_name + "_" + str(threshold), result)
		elif self.interval == "week":
			np.save("combination/combin_class_week_newdata_" + self.record_name + "_" + str(threshold), result)

def main():
	global total_num
	global total_correct
	total_num = 0
	total_correct = 0

	'''classifier_model = Classifier_Model(testing_time_id = 23, add_data = True, 
		add_date_feature = False, total_epoch = 30, record_name = "23_first", 
		model = 'nn', combination = True)'''
	
		#self.fout.write("pred, real, abs_error, squ_error \n") 
	'''for time_id in range(10, 103, 1):
		if time_id == 36: continue
		classifier_model = Classifier_Model(testing_time_id = time_id, add_data = True, 
			add_date_feature = True, total_epoch = 70, record_name = str(time_id) + "_first", 
			model = 'nn', combination = False, interval = "week", loss = "hinge")

	fout = open('record_class_week_10_102' + "", 'a')
	fout.write("\nAverage: ")
	fout.write("#(Correct) = %d, #(Total) = %d" %(total_correct, total_num))
	fout.write(", acc = %.4f\n" %(total_correct/float(total_num)))
	fout.close()'''

	for time_id in range(10, 103, 1):
		if time_id == 36: continue
		classifier_model = Classifier_Model(testing_time_id = time_id, add_data = True, 
			add_date_feature = True, total_epoch = 50, record_name = str(time_id) + "_first", 
			model = 'nn', combination = True, interval = "week")

	fout = open('record_class_week_10_102_newdata' + "", 'a')
	fout.write("\nAverage: ")
	fout.write("#(Correct) = %d, #(Total) = %d" %(total_correct, total_num))
	fout.write(", acc = %.4f\n" %(total_correct/float(total_num)))
	fout.close()

if __name__ == '__main__':
	main()

	
		