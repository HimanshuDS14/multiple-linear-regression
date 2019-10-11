import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv("50_Startups.csv")
print(data.head(10))


dummy = pd.get_dummies(data["State"] , drop_first=True)

data = pd.concat([dummy , data] , axis=1 )
print(data.head(10))

data.drop("State" , axis =1 , inplace = True)
print(data.head(10))

x = data.iloc[:,:-1].values
print(x)

y = data.iloc[:,[5]].values
print(y)

train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.3)

lin = LinearRegression()
lin.fit(train_x , train_y)


z = np.array([1,0,153441.51,101145.55 ,407934.54])
z = z.reshape(1,-1)

print(lin.predict(z))

score = r2_score(test_y , lin.predict(test_x))
print(score)

