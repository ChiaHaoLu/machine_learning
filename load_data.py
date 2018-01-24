import csv
import numpy as np
import tensorflow as tf
import sys

#data_size = int(sys.argv[1])
#y_dim = int(sys.argv[2])
#x_dim = int(sys.argv[3])
#text_path = sys.argv[4]

def load_data(data_size, x_dim, y_dim, text_path):
	data_y = []
	data_x = []
	for i in range(data_size):
		data_y.append([])
		data_x.append([])

	n_row = 0
	text = open(text_path, 'r') 
	row = csv.reader(text , delimiter=",")

	for r_data in row:
		if n_row != 0:
        		for i in range(x_dim+y_dim):
				if i < y_dim:
                			data_y[n_row-1].append(float(r_data[i]))
				else:
					data_x[n_row-1].append(float(r_data[i]))
		n_row = n_row+1
		if n_row > data_size :
			break
	text.close()

	data_x = np.array(data_x,dtype='float32')
	data_y = np.array(data_y,dtype='float32')
	print 'data x shape: ',data_x.shape
	print 'data y shape: ',data_y.shape
	return data_x, data_y
