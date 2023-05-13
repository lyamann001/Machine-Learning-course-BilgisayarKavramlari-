
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
df=pd.read_csv('../data/eksikveriler.csv')


print(df)

# row 12 and 16 yas nan
#sayisal veriler

#from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

age=df.iloc[:,1:4].values
print(age)

imputer=imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])
print(age)


#Label encoding

ulke=df.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

print("############################################")
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])

print(sonuc)



sonuc2=pd.DataFrame(data=age,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=df.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

#Farkli dataframe'leri birlestirmek icin kullanilir
concat1=pd.concat([sonuc,sonuc2],axis=1)
print(concat1)

concat2=pd.concat([concat1,sonuc3],axis=1)
print(concat2)



