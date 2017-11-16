# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:28:08 2017
an simple mode for nn
@author: 45183
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing

def add_layer(input, input_size,output_size,activate_function=None):
    #INTAL
    Weights = tf.Variable(tf.random_normal([input_size,output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
     # w*x+b
    wxb= tf.matmul(input, Weights) + biases
    if activate_function is None:
        return wxb
    else:
       return activate_function( wxb)



rawData=pd.read_csv("traindata.csv", header= None)
X=rawData.values
X=preprocessing.scale(X)
#print rawData
#row0 is id？
rawLabel=pd.read_csv("trainlabel.csv", header= None)
y=rawLabel.values

# input data
#we have 57 features and number of data is not constrained
x_place = tf.placeholder(tf.float32, [None, 57])
y_place = tf.placeholder(tf.float32, [None, 1])

# add hidden layer 
a=57
b=57
hidden1 = add_layer(x_place, 57, a, activate_function=tf.nn.relu)
hidden2 =add_layer(hidden1, a, b, activate_function=tf.nn.relu)


##predict
predict = add_layer(hidden2, b, 1, activate_function = tf.sigmoid)

##if wrong square=a constant  or 0
#reduction_indices = [1] for use the mean fuction
loss = tf.reduce_mean(tf.square(y_place - predict))


#train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(0.03).minimize(loss)

#initialize
init = tf.initialize_all_variables()

sess = tf.Session()
#run
sess.run(init)

for i in range(2000):
    sess.run(train_step, feed_dict = {x_place: X, y_place: y})
    if i % 200 == 0:
        print(sess.run(loss, feed_dict = {x_place: X, y_place: y}))

X=X.astype('float32')
result=sess.run(predict, feed_dict = {x_place: X})

error_rate=np.mean(np.abs(y-result))
print(error_rate)