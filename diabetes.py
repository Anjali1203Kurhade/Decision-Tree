
#importing library 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt

import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
#imprting dataset
dia=pd.read_csv("E:/data_science/Decision tree/Diabetes.csv")

dia.columns
dia[' Class variable'].unique()
dia[' Class variable'].value_counts()
colnames=list(dia.columns)

dia.dtypes
#predector
pred=colnames[0:8]
#target
tar=colnames[8]
#train,test split 
train,test=train_test_split(dia,test_size=0.2)
train[pred].dtypes



# Decision Tree
dt1=dt(criterion = 'entropy')
#fitting the data
dt1.fit(train[pred],train[tar])
#predecting on test data
pred_test=dt1.predict(test[pred])
#predecting on train data
pred_train=dt1.predict(train[pred])
#creating confusion matrix
pd.crosstab(test[tar],pred_test,rownames=['actual'],colnames=['predict'])
pd.crosstab(train[tar],pred_train,rownames=['actual'],colnames=['predict'])
#accuracy for test 
np.mean(test[tar]==pred_test)
#accuracy for train 
np.mean(train[tar]==pred_train)

# ramdom forest
r1=rfc(criterion='entropy')
#fitting the data
r1.fit(train[pred],train[tar])
#predecting on test data
pred_rfc_test=r1.predict(test[pred])
#predecting on train data
pred_rfc_train=r1.predict(train[pred])
#creating confusion matrix
pd.crosstab(test[tar],pred_rfc_test,rownames=['actual'],colnames=['predict'])
pd.crosstab(train[tar],pred_rfc_train,rownames=['actual'],colnames=['predict'])
#accuracy for test 
np.mean(pred_rfc_test==test[tar])
#accuracy for train
np.mean(pred_rfc_train==train[tar])
