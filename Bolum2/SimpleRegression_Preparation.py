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
veriler = pd.read_csv('../data/satislar.csv')
#pd.read_csv("veriler.csv")
#test

#VERI BOLME
aylar=veriler[['Aylar']] #aylar bagimsiz degisken
print(aylar)

satislar=veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,0:1].values #bağımli değişkenler
print(satislar2)

######################
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


#verilerin olceklenmesi
# from sklearn.preprocessing import StandardScaler
#
# sc=StandardScaler()
#
# X_train = sc.fit_transform(x_train)
# X_test = sc.transform(x_test)
# Y_train=sc.fit_transform(y_train)
# Y_test=sc.transform(y_test)


#Model Yaradiriq
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
#modeli built edir, insa edir . x train bagimli ve bagimsiz degiskenlerden olusuyor
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

print(tahmin)

#Gorsellestirme

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara gore satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
plt.show()
















