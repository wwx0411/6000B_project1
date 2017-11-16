# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:28:08 2017

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
    
    for i in range(2000):
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
X=preprocessing.scale(X)
#print rawData
#row0 is id？
rawLabel=pd.read_csv("trainlabel.csv", header= None)
y=rawLabel.values
a=57
b=57
#output=nn2(X,y,a,b,X,tf.nn.relu,tf.nn.relu,tf.sigmoid)



from sklearn.cross_validation import KFold
kfold=KFold(n=len(y), n_folds=5, shuffle=True)
for trainIndex, testIndex in kfold:
    print('')


c=[tf.nn.relu,None,tf.sigmoid]
#c=[tf.nn.relu,tf.sigmoid]

a_total=list([])
b_total=list([])
c1_total=list([])
c2_total=list([])
c3_total=list([])
error_rate=list([])
error_rate_max_min=list([])
for ia in range(1,2):
    a=ia+63-1
    for ib in range(1,2):
        
        b=ib+68-1
        print('a=',a,'b=',b)
        for c1 in c:
            for c2 in c:
                #for c3 in [tf.sigmoid]:
                for c3 in c:
                    print(" ")
                    a_total.append(a)
                    b_total.append(b)
                    c1_total.append(c1)
                    c2_total.append(c2)
                    c3_total.append(c3)
                    error_sum=list([])
                    """                                                            
                    for trainIndex, testIndex in kfold:
                        train_x,train_y=X[trainIndex], y[trainIndex]
                        test_x, test_y=X[testIndex], y[testIndex]
                        output=nn2(train_x,train_y,a,b,test_x,c1,c2,c3)
                        error_local=np.mean(np.abs(output-test_y))
                        error_sum.append(error_local)
                        """
                    train_x,train_y=X[trainIndex], y[trainIndex]
                    test_x, test_y=X[testIndex], y[testIndex]
                    
                    output=nn2(train_x,train_y,a,b,test_x,c1,c2,c3)
                    error_local=np.mean(np.abs(output-test_y))
                    error_sum.append(error_local)      
                        
                    error_sum=np.array(error_sum)
                    error_ave=np.mean(error_sum)
                    max_min=max(error_sum)-min(error_sum)
                    error_rate.append(error_ave)
                    error_rate_max_min.append(max_min)


for i in range(len(error_rate)):
    if error_rate[i]==min(error_rate):
        print(i)

'''   
#test not important and may be some wrong          
error_rate_c3_None=list([])
for i in range(len(error_rate)):
    if c3_total[i]==None:
        error_rate_c3_None.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c3_None)))
error_rate_c3_relu=list([])
for i in range(len(error_rate)):
    if c3_total[i]==tf.nn.relu:
        error_rate_c3_relu.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c3_relu)))
'''
'''
error_rate_c3_sigmoid=list([])
for i in range(len(error_rate)):
    if c3_total[i]==tf.nn.sigmoid:
        error_rate_c3_sigmoid.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c3_sigmoid)))



error_rate_c2_None=list([])
for i in range(len(error_rate)):
    if c2_total[i]==None:
        error_rate_c2_None.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c2_None)))
error_rate_c2_relu=list([])
for i in range(len(error_rate)):
    if c2_total[i]==tf.nn.relu:
        error_rate_c2_relu.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c2_relu)))

error_rate_c2_sigmoid=list([])
for i in range(len(error_rate)):
    if c2_total[i]==tf.nn.sigmoid:
        error_rate_c2_sigmoid.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c2_sigmoid)))


error_rate_c1_None=list([])
for i in range(len(error_rate)):
    if c1_total[i]==None:
        error_rate_c1_None.append(error_rate[i])
     
print(np.mean(np.array(error_rate_c1_None)))
error_rate_c1_relu=list([])
for i in range(len(error_rate)):
    if c1_total[i]==tf.nn.relu:
        error_rate_c1_relu.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c1_relu)))

error_rate_c1_sigmoid=list([])
for i in range(len(error_rate)):
    if c1_total[i]==tf.nn.sigmoid:
        error_rate_c1_sigmoid.append(error_rate[i])
        
print(np.mean(np.array(error_rate_c1_sigmoid)))



#print(min(error_rate_c3_None))
#print(min(error_rate_c3_relu))
print(min(error_rate_c3_sigmoid))
print(min(error_rate_c2_None))
print(min(error_rate_c2_relu))
print(min(error_rate_c2_sigmoid))
print(min(error_rate_c1_None))
print(min(error_rate_c1_relu))
print(min(error_rate_c1_sigmoid))
'''

