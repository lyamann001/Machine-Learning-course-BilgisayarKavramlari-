
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


