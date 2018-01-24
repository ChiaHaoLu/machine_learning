import csv
import numpy as np
from tf_rbf import TFRBFNet as RBFNet
import load_data as ld

data_size = 354694
#data_size = 10
x_dim = 34
y_dim = 1
train_x,train_y = ld.load_data(data_size,x_dim,y_dim,"data/train.csv")

#train_x = train_x[0:10]
#train_y = train_y[0:10]
#print train_x
rbf = RBFNet(k=10)
rbf.fit(train_x, train_y)

prediction = rbf.predict(train_x[1:100])
print prediction
