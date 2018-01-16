import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt
import load_data as ld
import tensorflow as tf

from keras.datasets import mnist
from keras.utils import np_utils,to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import PReLU as PRelu
from keras.optimizers import SGD,Adagrad,Adam,Nadam
data_size = 710315
x_dim = 3
y_dim = 1

train_x,train_y = ld.load_data(data_size,x_dim,y_dim,"data/train.csv")

file = open("model_data.csv","wb")
csv_w = csv.writer(file)

for i in range(3,6):
	for j in range(0,10):
		model = Sequential()
		model.add(Dense(i,input_shape = (x_dim,),  kernel_initializer='glorot_uniform',activation=PRelu(alpha_initializer='zeros')))
		model.add(Dense(y_dim,input_shape = (i,),  kernel_initializer='glorot_uniform'))

		model.compile(loss='mse', optimizer = Adam(lr=1) )
		model.fit(train_x,train_y, batch_size = 100, epochs = 3)
		score = model.evaluate(train_x,train_y)
		csv_w.writerow([i,score, model.get_weights()])
#model.compile(loss='mse', optimizer = Nadam(lr=0.1) )
#predicted_m = model.get_weights()[0]
#predicted_b = model.get_weights()[1]
#print predicted_m, predicted_b

#score = model.evaluate(test_x,test_y)
#print "\nTrain Acc:", score
#print model.get_weights()
#print "\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b)
#print model.predict(data_x)

file.close()
