#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






# veri yukleme
veriler = pd.read_csv('../data/maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values #numpy array
Y = y.values #numpy array
print(X)
#no need
# from sklearn.preprocessing import StandardScaler
# sc1=StandardScaler()
# x_olcekli=sc1.fit_transform(X)
# sc2=StandardScaler()
# y_olcekli=np.ravel(sc2.fit_transform(Y.reshape(-1,1)))



# from sklearn.tree import DecisionTreeRegressor
# r_dt=DecisionTreeRegressor(random_state=0)
# r_dt.fit(X,Y)
# plt.scatter(X,Y,color='red')
# plt.plot(X,r_dt.predict(X),color='blue')
# plt.show()
#
# print(r_dt.predict([[11]]))
# print(r_dt.predict([[6.6]]))


from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)  # X den Y ni ogren
plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')

plt.show()

print(r_dt.predict([[15]]))
