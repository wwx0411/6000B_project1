# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 12:57:59 2017

@author: 45183
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
rawData=pd.read_csv("traindata.csv", header= None)
X=rawData.values
X=preprocessing.scale(X)
#print rawData
#row0 is id？
rawLabel=pd.read_csv("trainlabel.csv", header= None)
y=rawLabel.values
##print X
##print len(y)

##Normalization
## Subtract the mean for each feature
##Divide each feature by its standard deviation
##！！！AFTER DOING THAT, SOME CLASSIFER CAN NOT BE USED
#print X
#X -= np.mean(X, axis=0)
#X /= np.std(X, axis=0)
#print X

##import model
from sklearn.svm import SVC
SVM=SVC();
SVM2= SVC(kernel='linear')##to long

from sklearn.linear_model import LogisticRegression
IR= LogisticRegression()

from sklearn.lda import LDA
LDA=LDA()

from sklearn.qda import QDA
QDA=QDA()
 
from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier=RandomForestClassifier()

from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import AdaBoostClassifier


model=[SVM,IR,LDA,QDA,Tree,GaussianNB(),AdaBoostClassifier(),Perceptron()]

#5-fold-cv
from sklearn.cross_validation import KFold
kfold=KFold(n=len(y), n_folds=5, shuffle=True)
#kfold=KFold(n=len(y), n_folds=5, shuffle=False)
for m in model:
    print (str(m).split('(')[0])
    sum_testAccurancy=0
    for trainIndex, testIndex in kfold:
       train_x,train_y=X[trainIndex], y[trainIndex]
       test_x, test_y=X[testIndex], y[testIndex]
       m.fit(train_x,train_y)
       trainAccurancy=np.mean(m.predict(train_x)==train_y)
       testAccurancy=np.mean(m.predict(test_x)==test_y)
       sum_testAccurancy=sum_testAccurancy+testAccurancy
       print ("Train Accurancy:%f,    Test Accurancy:%f" %(trainAccurancy,testAccurancy))
    ave_testAccurancy=sum_testAccurancy/5
    print ("Average Test Accurancy:%f" %(ave_testAccurancy))
    print ('  ')
    
