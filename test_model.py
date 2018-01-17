import csv
import numpy as np
from tf_rbf import TFRBFNet as RBFNet
import load_data as ld

data_size = 710315
x_dim = 3
y_dim = 1
train_x,train_y = ld.load_data(data_size,x_dim,y_dim,"data/train.csv")

rbf = RBFNet(k=10)
rbf.fit(train_x, train_y)

prediction = rbf.predict(train_x)
#print prediction
