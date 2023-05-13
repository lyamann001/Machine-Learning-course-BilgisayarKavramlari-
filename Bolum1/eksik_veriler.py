import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer #missing value tamamlamaq ucun
#veri yukleme
veriler=pd.read_csv('../data/eksikveriler.csv')
print(veriler)

# boy=veriler[['boy']]
# print(boy)

# boykilo=veriler[['boy','kilo']]
# print(boykilo)

#missing values (eksik values)
#creating object
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=veriler.iloc[:,1:4].values
print(yas)
yas[:,1:4]=imputer.fit_transform(yas[:,1:4])
print(yas)

#second way:
# imputer=imputer.fit(yas[:,1:4])
# yas[:,1:4]=imputer.transform(yas[:,1:4])
# print(yas)

