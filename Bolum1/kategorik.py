import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer #missing value tamamlamaq ucun
#veri yukleme
from sklearn import preprocessing
veriler=pd.read_csv('../data/eksikveriler.csv')
# print(veriler)

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=veriler.iloc[:,1:4].values
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

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

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
cinsiyet=veriler.iloc[:,-1].values

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1) #2 data frame i birlsedirir
s2=pd.concat([s,sonuc3],axis=1)
print(s2)




