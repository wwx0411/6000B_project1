# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:21:05 2017

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


def nn2(train_input,train_output,a,b,test_input,activate_function1,activate_function2,activate_function3):    
   # rawData=pd.read_csv("traindata.csv", header= None)
    X=train_input
    X=preprocessing.scale(X)
    #print rawData
    #row0 is id？
    #rawLabel=pd.read_csv("trainlabel.csv", header= None)
    y=train_output
    
    # input data
    #we have 57 features and number of data is not constrained
    x_place = tf.placeholder(tf.float32, [None, 57])
    y_place = tf.placeholder(tf.float32, [None, 1])
    
    # add hidden layer 

    hidden1 = add_layer(x_place, 57, a, activate_function=activate_function1)
    hidden2 =add_layer(hidden1, a, b, activate_function=activate_function2)
    
    
    ##predict
    predict = add_layer(hidden2, b, 1, activate_function = activate_function3)
    
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
    
    for i in range(3000):
        sess.run(train_step, feed_dict = {x_place: X, y_place: y})
        if i % 200 == 0:
            print(sess.run(loss, feed_dict = {x_place: X, y_place: y}))
    
    #output
    test_input=preprocessing.scale(test_input)
    result=sess.run(predict, feed_dict = {x_place: test_input})
    result=[0 if i <0.5 else 1 for i in result]
    #make result in one col
    result=np.array([result])
    result=result.reshape(np.size(result),1)
    sess.close()
    return result


rawData=pd.read_csv("traindata.csv", header= None)
X=rawData.values
train_x=preprocessing.scale(X)
#print rawData
#row0 is id？
rawLabel=pd.read_csv("trainlabel.csv", header= None)
train_y=rawLabel.values


a=63
b=68
c1=c2=c3=tf.sigmoid

rawLabel=pd.read_csv("testdata.csv", header= None)
test_x=rawLabel.values

predict=nn2(train_x,train_y,a,b,test_x,c1,c2,c3)

np.savetxt('project1_20476516.csv', predict, delimiter = ',')

