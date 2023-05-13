import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer #missing value tamamlamaq ucun
#veri yukleme
from sklearn import preprocessing
veriler=pd.read_csv('../data/veriler.csv')
# print(veriler)

yas=veriler.iloc[:,1:4].values

#label encoding process
#it gives us only numerical variable
ulke=veriler.iloc[:,0:1].values
print(ulke)
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print("Label Encoder")
print(ulke)

#one hot encoding
#after converted to numerical variables
#One-hot encoding is the representation of categorical variables as binary vectors.

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print("One HOT ENCODER")
print(ulke)


ulke=veriler.iloc[:,0:1].values
print(ulke)
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print("Label Encoder")
print(ulke)

#one hot encoding
#after converted to numerical variables
#One-hot encoding is the representation of categorical variables as binary vectors.

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print("One HOT ENCODER")
print(ulke)

######################################

print("cinsiyet")
c=veriler.iloc[:,-1:].values
print(c)
le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(c[:,-1])
print("Label Encoder Cinsiyyet")
print(c)

#one hot encoding
#after converted to numerical variables
#One-hot encoding is the representation of categorical variables as binary vectors.

ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print("One HOT ENCODER Cinsiyyet")
print(c)


sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
cinsiyet=veriler.iloc[:,-1].values

sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1) #2 data frame i birlsedirir
s2=pd.concat([s,sonuc3],axis=1)
print(s2)


from sklearn.model_selection import train_test_split


x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train) #aralarinda model kurmak

y_pred=regressor.predict(x_test)

print(y_pred)
print(y_test)

boy=s2.iloc[:3:4].values
print(boy)
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]

veri=pd.concat([sol,sag],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)

#Backward Elimination

import statsmodels.api as sm

#degiskenleri sisteme eklemek

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)

model=sm.OLS(boy,X_l).fit()

print("Model")
print(model.summary())



#Multi Linear Regression icinde sehvlik var . Gerileme de butun kodlar duz yazilib
