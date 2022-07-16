# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:36:59 2022

@author: ANJALI
""" 
#importing library 
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rfc
#importing library 
fraud=pd.read_csv("E:/data_science/Decision tree/Fraud_check.csv")
fraud.columns
# for taxabble income less than 30001 are risky represented  as 0 and those who are ahving taxable income greater than 30000 are termed as good represented  as 1 
fraud['Taxable.Income']=np.where(fraud['Taxable.Income'] <= 30000, 0 ,fraud['Taxable.Income'])
fraud['Taxable.Income']=np.where(fraud['Taxable.Income'] > 30000, 1 ,fraud['Taxable.Income'])
fraud=fraud.iloc[:,[2,0,1,3,4,5]]
colnames=list(fraud.columns)
#predictor
pred=colnames[1:]
#target
tar=colnames[0]
#preprocessing
le=LabelEncoder()
fraud['Undergrad']=le.fit_transform(fraud['Undergrad'])
fraud['Marital.Status']=le.fit_transform(fraud['Marital.Status'])
fraud['Urban']=le.fit_transform(fraud['Urban'])
train,test=train_test_split(fraud,test_size=0.25)

#######################################################
#Decison tree Classifire
dt1=dt( criterion='entropy')
#fitting the model
dt1.fit(train[pred],train[tar])
#predecting on test data
test_pred=dt1.predict(test[pred])
#Confusion matrix
pd.crosstab(test_pred, test[tar], colnames=['actual'],rownames=['predicted'])
#prediction of train data
train_pred=dt1.predict(train[pred])
#Calculating accuracy
np.mean(test_pred == test[tar])
np.mean(train_pred == train[tar])


##########################################################
#random forest Classifier
rf=rfc(n_estimators=28,criterion='entropy')
#fitting the model
rf.fit(train[pred],train[tar])
#predecting on test data
pred_rf_test=rf.predict(test[pred])

#Confusion matrix
pd.crosstab(pred_rf_test,test[tar], colnames=['actual'],rownames=['predicted'])
#Calculating accuracy
np.mean(pred_rf_test == test[tar])

