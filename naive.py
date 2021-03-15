# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:54:33 2021

@author: Mazen
"""


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df=pd.read_csv("C:\\Users\\Mazen\\Downloads\\Bio-Assignment-2\\fetal_health.csv")
y=df.iloc[:,-1:]
df = df.drop(['fetal_health'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df,y , test_size=0.3, random_state=8)
x_test.to_csv('TestData.csv',index=False)
data2 = pd.read_csv("TestData.csv")
nv = GaussianNB() 
nv.fit(df,y.values.ravel()) 
y_pred = nv.predict(data2) 
p=accuracy_score(y_test,y_pred) *100
print(p)

