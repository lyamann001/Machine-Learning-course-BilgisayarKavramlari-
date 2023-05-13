import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("../data/maaslar.csv")

print(veriler)
#data frame slice
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#Numpy array donusumu
X=x.values
Y=y.values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()
#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
poly_reg=PolynomialFeatures(degree=4)

x_poly=poly_reg.fit_transform(X)
#print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()


#tahminledr


print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
