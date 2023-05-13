# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('../data/odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
#print(veriler)

#encoder: Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
veriler2=veriler.apply(LabelEncoder().fit_transform)

print(veriler2)


from sklearn.preprocessing import OneHotEncoder\

c=veriler2.iloc[:,:1].values
ohe=OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(14),columns=['o','r','s'])

sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)

sonveriler=pd.concat([sonveriler,veriler2.iloc[:,-2:]],axis=1)

print(sonveriler)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols= sm.ols(endog=sonveriler.iloc[:,-1:],exog=X_l)
r=r_ols.fit()
print(r.summary())






