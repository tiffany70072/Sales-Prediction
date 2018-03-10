# 12/22/2017
# Original file name: preprocess_classifier_1222
# Prepare the data for binary classifier
# add all artificial features here

#import tensorflow as tf
import numpy as np 
import pandas as pd
import random

'''from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config = config))'''

class Model(object):
	def __init__(self, interval = "month"):
		self.interval = interval
		if self.interval == "month": path = "data-112017/"
		elif self.interval == "week": path = "data-122017-week/"
		self.load(path)

		self.dim = 4 + 1 + 8 + 4 + 8 # basic information, real value (zero or not), features 
		#self.data = np.zeros([np.sum(self.exist_first[:, :, 1:]), self.dim], dtype = float)
		self.data = np.zeros([np.sum(self.exist[:, :, 1:]), self.dim], dtype = float)
		print "data.shape =", self.data.shape
		self.add_basic()
		#self.store_previous_id()
		self.compute_average()
		self.add_features()
		#np.save("data-112017/classifier_23", self.data)
		print self.data.shape
		np.save("data-122017-week/classifier_week_all", self.data)

	def load(self, path):
		print "Loading"
		self.amount = np.load(path + "amount.npy")
		#self.amount = np.load(path + "number.npy")				# sales_amount or sales_count
		self.price 	= np.load(path + "price.npy")
		self.exist 	= np.load(path + "exist.npy") 				# 1: data exist, 0: NIL
		self.exist_first = np.load(path + "exist_first.npy") 	# 1: data exist first time, 0: others
		self.first_zero	 = np.load(path + "first_zero.npy") 	# 1: data exist first time and zero, 0: others
			
		print "#(Data) =", np.sum(self.exist)
		print "#(Exist first) =", np.sum(self.exist_first)

	def output_data(self, n, data = None):
			if data == None: data = self.data[:n]
			print "device, product, month, real, price,", 
			print "previous amount: (product/device), (all/first), (one/all)",
			print "times: (product/device), (all/first),",
			print "ratio(!=zero) first time: (product/device), (one/all)"
			for i in range(n): 
				print "%d\t%d\t%d\t%d\t" %(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])),
				print "%.2f\t" % data[i][4],
				for j in range(5, 25): print "%.1f\t" %int(data[i][j]),
				print ""

	def add_basic(self): 
		count = 0
		for i in range(self.exist.shape[0]):
			for j in range(self.exist.shape[1]):
				for k in range(1, self.exist.shape[2]):
					#if self.exist_first[i][j][k] == 1:
					if self.exist[i][j][k] == 1:
						self.data[count][0] = i 		# device_id
						self.data[count][1] = j			# product_id
						self.data[count][2] = k 		# month
						if self.amount[i][j][k] != 0: self.data[count][3] = 1 # amount != zero
						else: self.data[count][3] = 0
						self.data[count][4] = self.price[i][j][k]

						count += 1
						if count % 100000 == 0: print count
		print "Count =", count, np.sum(self.exist_first[:, :, 1:])

	def store_previous_id(self):
		count_missing_all_one = 0 # same product, data in other device is missing in previous one month
		count_missing_all_all = 0
		count_missing_first_one = 0 # exist the first time
		count_missing_first_all = 0
		
		product_all_one = [] 	# store id for previous one month
		product_all_all = []	# store id for previous all times
		product_first_one = [] 	# exist the first time
		product_first_all = []

		print "\nStore previous id..."
		for count in range(self.data.shape[0]):
			if count % 1000 == 0: print count
			i0, j0, k0 = int(self.data[count][0]), int(self.data[count][1]), int(self.data[count][2])
			temp_all   = []
			temp_first = []
			for i in range(self.exist.shape[0]):
				if self.exist[i][j0][k0-1] == 1: 
					temp_all.append([i, j0, k0-1])
				if self.exist_first[i][j0][k0-1] == 1:
					temp_first.append([i, j0, k0-1])
			product_all_one.append(temp_all)
			product_first_one.append(temp_first)
			if len(temp_all)   == 0: count_missing_all_one += 1
			if len(temp_first) == 0: count_missing_first_one += 1
		print "List.len =", len(product_all_one), len(product_first_one)
		#print product_all_one[0:10]
		#print product_first_one[0:10]
		print "Count missing = (all)", count_missing_all_one, ", (first)", count_missing_first_one
				
	def compute_average(self):
		print "\nStart computing average..."
		
		# compute the average amount for each month
		if self.interval == "month": self.testing_time = 23
		elif self.interval == "week": self.testing_time = 103 - 1
		self.average_time = np.zeros([self.testing_time], dtype = float)
		self.average_time_first = np.zeros([self.testing_time], dtype = float)
		for k in range(self.testing_time):
			total_amount = 0
			total_amount_first = 0
			for i in range(self.exist.shape[0]):
				for j in range(self.exist.shape[1]):
					if self.exist[i][j][k] != 0: 	   total_amount += self.amount[i][j][k]
					if self.exist_first[i][j][k] != 0: total_amount_first += self.amount[i][j][k]
			self.average_time[k] = total_amount/float(np.sum(self.exist[:, :, k]))
			if np.sum(self.exist_first[:, :, k]) != 0:
				self.average_time_first[k] = total_amount_first/float(np.sum(self.exist_first[:, :, k]))
			else: self.average_time_first[k] = 0		
		self.average_all = np.mean(self.average_time)
		self.average_all_first = np.mean(self.average_time_first)

		print "Average amount for whole data =", self.average_all
		print "Average amount for each month =\n", 
		for i in range(self.testing_time): print "%.1f" %self.average_time[i],
		print ""

		print "Average amount for data at the first time =", self.average_all_first
		print "Average amount for data at the first time of each month =\n", 
		for i in range(self.testing_time): print "%.1f" %self.average_time_first[i],
		print ""

		# compute the average amount for one "product" in one month for every device
		self.average_product = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.count_device = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.average_product_first = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.count_device_first = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.cumu_average_product = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.cumu_count_device = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.cumu_average_product_first = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		self.cumu_count_device_first = np.zeros([self.exist.shape[1], self.testing_time], dtype = float)
		count_missing = 0 # if there is no data in other device
		count_missing_first = 0 # if there is no data in other device
		for j in range(self.exist.shape[1]):
			for k in range(self.testing_time):
				total = 0
				total_first = 0
				for i in range(self.exist.shape[0]): # device
					if self.exist[i][j][k] == 1: total += self.amount[i][j][k]
					if self.exist_first[i][j][k] == 1: total_first += self.amount[i][j][k]

				self.count_device[j][k] = np.sum(self.exist[:, j, k])
				if self.count_device[j][k] == 0: count_missing += 1
				else: 
					self.average_product[j][k] = total/float(self.count_device[j][k])
					self.cumu_average_product[j][k] += total
					self.cumu_count_device += self.count_device[j][k]

				self.count_device_first[j][k] = np.sum(self.exist_first[:, j, k])
				if self.count_device_first[j][k] == 0: count_missing_first += 1
				else: 
					self.average_product_first[j][k] = total_first/float(self.count_device_first[j][k])
					self.cumu_average_product_first[j][k] += total_first
					self.cumu_count_device_first += self.count_device_first[j][k]
					
		print "#(No previous data for specific product)", count_missing
		print "#(No previous data for specific product in the first time)", count_missing_first
		print "Whole Number (product * date) =", self.exist.shape[1]*self.testing_time

		# compute the average amount for one "product" in one month for every device
		self.average_device = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.count_product = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.average_device_first = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.count_product_first = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.cumu_average_device = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.cumu_count_product = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.cumu_average_device_first = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		self.cumu_count_product_first = np.zeros([self.exist.shape[0], self.testing_time], dtype = float)
		count_missing = 0 # if there is no data for other products
		count_missing_first = 0 # if there is no data for other products
		for i in range(self.exist.shape[0]):
			for k in range(self.testing_time):
				total = 0
				total_first = 0
				for j in range(self.exist.shape[1]): # device
					if self.exist[i][j][k] == 1: total += self.amount[i][j][k]
					if self.exist_first[i][j][k] == 1: total_first += self.amount[i][j][k]

				self.count_product[i][k] = np.sum(self.exist[i, :, k])
				if self.count_product[i][k] == 0: count_missing += 1
				else: 
					self.average_device[i][k] = total/float(self.count_product[i][k])
					self.cumu_average_device[i][k] += total
					self.cumu_count_product += self.count_product[i][k]


				self.count_product_first[i][k] = np.sum(self.exist_first[i, :, k])
				if self.count_product_first[i][k] == 0: count_missing_first += 1
				else: 
					self.average_device_first[i][k] = total_first/float(self.count_product_first[i][k])
					self.cumu_average_device_first[i][k] += total_first
					self.cumu_count_product += self.count_product[i][k]
					
		print "#(No previous data for specific device)", count_missing
		print "#(No previous data for specific device in the first time)", count_missing_first
		print "Whole Number (device * date) =", self.exist.shape[0]*self.testing_time
	
	def add_features(self):
		print "\nAdd features..."
		self.count_different = [0, 0]

		self.count_missing = np.zeros([25], dtype = int)

		for count in range(self.data.shape[0]):
			if count % 10000 == 0: print count
			i0, j0, k0 = int(self.data[count][0]), int(self.data[count][1]), int(self.data[count][2])
			
			# add previous amount for each combination properties
			self.add_previous_amount(count, i0, j0, k0)
			
			# add how many times of products have been sold
			self.add_sales_times(count, i0, j0, k0)

			# add if amount = zero for each combination properties
			self.add_sold_out_ratio(count, i0, j0, k0)

		for i in range(5, 13):
			print "#(Missing) =", self.count_missing[i]
		self.output_data(10)

	def add_previous_amount(self, count, i0, j0, k0):
		# previous amount for the same product in other devices 
		# amount for previous one month
		if self.count_device[j0][k0-1] != 0: self.data[count][5] = self.average_product[j0][k0-1]
		else: 
			self.data[count][5] = self.average_time[k0-1] # or zero?
			self.count_missing[5] += 1
			
		# amount for previous all times
		exist_month = 0
		total = 0
		for k in range(k0):
			if self.count_device[j0][k] != 0: 
				total += self.average_product[j0][k]
				exist_month += 1
		if exist_month != 0: self.data[count][6] = total/float(exist_month)
		else: 
			self.data[count][6] = self.data[count][5]
			self.count_missing[6] += 1
		if self.data[count][6] != self.data[count][5]: self.count_different[0] += 1
		
		# amount for previous one month, for the first time product
		if self.count_device_first[j0][k0-1] != 0: self.data[count][7] = self.average_product_first[j0][k0-1]
		else: 
			self.data[count][7] = 0 # or zero?
			self.count_missing[7] += 1

		# amount for previous all times, for the first time product
		exist_month = 0
		total = 0
		for k in range(k0):
			if self.count_device_first[j0][k] != 0: 
				total += self.average_product_first[j0][k]
				exist_month += 1
		if exist_month != 0: self.data[count][8] = total/float(exist_month)
		else: 
			self.data[count][8] = 0 # check here
			self.count_missing[8] += 1

		# previous amount for other products in the same device
		# amount for previous one month
		if self.count_product[i0][k0-1] != 0: self.data[count][5] = self.average_device[i0][k0-1]
		else: 
			self.data[count][9] = self.average_time[k0-1] # or zero?
			self.count_missing[9] += 1
			
		# amount for previous all times
		exist_month = 0
		total = 0
		for k in range(k0):
			if self.count_product[i0][k] != 0: 
				total += self.average_device[i0][k]
				exist_month += 1
		if exist_month != 0: self.data[count][10] = total/float(exist_month)
		else: 
			self.data[count][10] = self.data[count][9]
			self.count_missing[10] += 1
		if self.data[count][10] != self.data[count][9]: self.count_different[1] += 1
		
		# amount for previous one month, for the first time product
		if self.count_product_first[i0][k0-1] != 0: self.data[count][11] = self.average_device_first[i0][k0-1]
		else: 
			self.data[count][11] = 0 # or zero?
			self.count_missing[11] += 1

		# amount for previous all times, for the first time product
		exist_month = 0
		total = 0
		for k in range(k0):
			if self.count_product_first[i0][k] != 0: 
				total += self.average_device_first[i0][k]
				exist_month += 1
		if exist_month != 0: self.data[count][12] = total/float(exist_month)
		else: 
			self.data[count][12] = 0 # check here
			self.count_missing[12] += 1

	def add_sales_times(self, count, i0, j0, k0):
		# add how many times of same product have been sold in all devices
		self.data[count][13] = np.sum(self.exist[:, j0, :k0])
		self.data[count][14] = np.sum(self.exist_first[:, j0, :k0])
		# add how many times of all products have been sold in same device
		self.data[count][15] = np.sum(self.exist[i0:, :, :k0])
		self.data[count][16] = np.sum(self.exist_first[i0:, :, :k0])

	def add_sold_out_ratio(self, count, i0, j0, k0):
		# add whether different --> count
		# same product, one month, all
		
		# same product, all times, all
		
		# same product, one month, first time
		if np.sum(self.exist_first[:, j0, k0-1]) != 0:
			self.data[count][19] = 1 - np.sum(self.first_zero[:, j0, k0 - 1])/float(np.sum(self.exist_first[:, j0, k0 - 1]))
		else: self.data[count][19] = 0
		# same product, all times, first time
		if np.sum(self.exist_first[:, j0, :k0]) != 0:
			self.data[count][20] = 1 - np.sum(self.first_zero[:, j0, :k0])/float(np.sum(self.exist_first[:, j0, :k0]))
		else: self.data[count][20] = 0
		
		# same device, one month, all
		
		# same device, all times, all
		
		# same device, one month, first time
		if np.sum(self.exist_first[i0, :, k0-1]) != 0:
			self.data[count][23] = 1 - np.sum(self.first_zero[i0, :, k0 - 1])/float(np.sum(self.exist_first[i0, :, k0 - 1]))
		else:
			self.data[count][23] = 0
		# same device, all times, first time
		if np.sum(self.exist_first[i0, :, :k0]) != 0:
			self.data[count][24] = 1 - np.sum(self.first_zero[i0, :, :k0])/float(np.sum(self.exist_first[i0, :, :k0]))
		else:
			self.data[count][24] = 0
		
def main():
	model = Model(interval = "week")

if __name__ == '__main__':
	main()


	
