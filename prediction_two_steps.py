# Method: combination of binary classifier and tensor factorization
# 12/04/2017
# Goal: predict the numbers or amounts of vending machine
# Combine two steps in this program
# Step 1: Classifier, predict zero or not
# Step 2: Tensor factorization, predict the value if it is not zero

import tensorflow as tf
import numpy as np 

from tensor_factorization import TensorFactorization			# model of tensor factorization
from classifier import Classifier_Model 	# model of classifier

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config = config))

class Combination(object):
	def __init__(self, target = "amount", testing_file = None, record_name = None, 
			interval = None):
		self.record_name = record_name
		self.testing_file = testing_file
		self.interval = interval

		self.first = True
		
		'''for threshold in [0.2, 0.5, 0.8, 0.9, 0.99]:
			self.load(target, testing_file, threshold)
			#self.predict_tf()
			self.predict(threshold)
			self.first = False'''

		for threshold in [0.5]:
			self.load(target, testing_file, threshold)
			self.predict_tf()
			self.predict(threshold)
			self.first = False

	def load(self, target, testing_file, threshold):
		if self.interval == "month":
			self.is_zero = np.load("combination/combin_class_" + self.testing_file + "_" + str(threshold) + ".npy")
			#self.prediction = np.load("combination/combin_tf_" + self.record_name + ".npy")
			self.prediction = np.load("combination/predict_" + self.testing_file + ".npy")
			path = "data-112017/"
		elif self.interval == "week":
			self.is_zero = np.load("combination/classifier_0102/combin_class_week_" + self.testing_file + "_first_" + str(threshold) + ".npy")
			self.prediction = np.load("combination/predict_Alicia/0103_predict_" + self.testing_file + "_week.npy")
			#self.prediction = np.load("combination/tf_0108/combin_tf_week_" + self.testing_file + ".npy")
			path = "data-122017-week/"
		
		if   target == "amount": self.real = np.load(path + "amount.npy") 	# data of sales amount
		elif target == "number": self.real = np.load(path + "number.npy") 	# data of sales item count
		else:
			print "Wrong target setting"
			exit()
		#self.test_list = np.load("data-112017/test_" + testing_file + ".npy")
		print testing_file
		'''if self.first == True:
			print "is_zero.shape =", self.is_zero.shape
			print self.is_zero[:5]
			print "prediction.shape =", self.prediction.shape
			print self.prediction[:5]'''
		#exit()

	def predict_tf(self):
		pred = []
		real = []
		abso = 0
		squa = 0
		for i in range(self.prediction.shape[0]):
			if self.prediction[i][3] < 0:
				pred.append(0)
			else: pred.append(self.prediction[i][3])
			#pred.append(self.prediction[i][3])
			#pred.append(0)
			real.append(self.real[int(self.prediction[i][0])][int(self.prediction[i][1])][int(self.prediction[i][2])])
			#print self.is_zero[i][3] == 0, self.prediction[i][3], pred[-1], real[-1]
			abso += abs(pred[-1] - real[-1])
			squa += (pred[-1] - real[-1])**2
		print "MAE =", abso/float(self.prediction.shape[0])
		print "MSE =", squa/float(self.prediction.shape[0])

	def predict(self, threshold):
		pred = []
		real = []
		abso = 0
		squa = 0
		self.is_zero_index = np.full((499, 433), -1)
		for i in range(self.is_zero.shape[0]):
			self.is_zero_index[self.is_zero[i][0]][self.is_zero[i][1]] = self.is_zero[i][3]
		
		#print "pred = 0, pred, pred, real"
		for i in range(self.prediction.shape[0]):
			if self.is_zero_index[int(self.prediction[i][0])][int(self.prediction[i][1])] == -1:
				print "error"
			elif self.is_zero_index[int(self.prediction[i][0])][int(self.prediction[i][1])] == 0: 
				pred.append(0)
			else: 
				if self.prediction[i][3] < 0:
					pred.append(0)
				else: pred.append(self.prediction[i][3])
			real.append(self.real[int(self.prediction[i][0])][int(self.prediction[i][1])][int(self.prediction[i][2])])
			#print self.is_zero_index[int(self.prediction[i][0])][int(self.prediction[i][1])] == 0, 
			#print "%.2f" %self.prediction[i][3], real[-1]
			abso += abs(pred[-1] - real[-1])
			squa += (pred[-1] - real[-1])**2
		print "\nthreshold =", threshold
		print "MAE =", abso/float(self.prediction.shape[0])
		print "MSE =", squa/float(self.prediction.shape[0]), "\n"

class PredictZero(object):
	def __init__(self, target = "amount", test_id = None, 
			interval = None):
		
		self.interval = interval
		self.load(target, test_id)
		self.predict(test_id)
		
	def load(self, target, test_id):
		if self.interval == "month": 
			path = "data-112017/"
			self.exist_first = np.load(path + "exist_first.npy")
		elif self.interval == "week": 
			path = "data-122017-week/"
			self.exist_first = np.load(path + "exist_first.npy")
		
		if   target == "amount": self.real = np.load(path + "amount.npy") 	# data of sales amount
		elif target == "number": self.real = np.load(path + "number.npy") 	# data of sales item count
		else:
			print "Wrong target setting"
			exit()

	def predict(self, test_id):
		pred = []
		real = []
		abso = 0
		squa = 0
		count = 0
		for i in range(self.real.shape[0]):
			for j in range(self.real.shape[1]):
				if self.exist_first[i][j][test_id] == 1:
					pred.append(0)
					real.append(self.real[i][j][test_id])
					abso += abs(pred[-1] - real[-1])
					squa += (pred[-1] - real[-1])**2
					count += 1
		print "Testing id =", test_id, ", count =", count
		print "MAE = %.2f" %(abso/float(count)),
		print "MSE = %.2f" %(squa/float(count)), "\n"

def main():
	target = "amount"

	for test_id in range(10, 103):
		if test_id == 36: continue # or test_id == 35 or (test_id >= 54 and test_id <= 58): continue
		testing_file = str(test_id)
		combination = Combination(target = target, testing_file = testing_file, 
			interval = "week")
		#model = PredictZero(target = target, test_id = test_id, interval = "week")
	exit()

	'''for month in range(21, 24):
		#month = 21
		testing_file = str(month) + "_first"
		record_name = "test"

		tensorfact = TensorFactorization(target = target, testing_file = testing_file, 
			record_name = record_name, combination = True,
			reuse_model = False, store_model = True, total_epoch = 6000)
		tensorfact = TensorFactorization(target = target, testing_file = testing_file, 
			record_name = record_name, combination = True,
			reuse_model = True, store_model = False, total_epoch = 1)
		
		classifier_model = Classifier_Model(testing_month = month, add_data = True, 
			add_date_feature = False, total_epoch = 30, record_name = testing_file, 
			model = 'nn', combination = True)
		
		combination = Combination(target = target, testing_file = testing_file, 
			record_name = record_name, interval = "week")'''


	for time_id in [28, 37, 29, 30, 31, 18, 19, 25, 74, 44, 45, 46, 54, 55, 56]:
		#month = 21
		if time_id == 36: continue
		print "week =", time_id
		testing_file = str(time_id)
		record_name = "test"
		'''tensorfact = TensorFactorization(target = target, testing_file = testing_file, 
			record_name = record_name, combination = True,
			reuse_model = False, store_model = True, total_epoch = 6000)
		tensorfact = TensorFactorization(target = target, testing_file = testing_file, 
			record_name = record_name, combination = True,
			reuse_model = True, store_model = False, total_epoch = 1)
		'''
		
		combination = Combination(target = target, testing_file = testing_file, 
			record_name = record_name, interval = "week")

if __name__ == '__main__':
	main()


		
