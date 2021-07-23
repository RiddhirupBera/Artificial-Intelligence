

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset_train = pd.read_csv('/home/riddhirup/Downloads/Restaurant_Reviews.tsv',delimiter="\t")
train_set = dataset_train.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scaled_training = sc.fit_transform(train_set)

x_train = []
y_train = []

for i in range(60,1258):
    x_train.append(scaled_training[i-60:i,0])
    y_train.append(scaled_training[i,0])
    
x_train,y_train = np.array(x_train),np.array(y_train)  
#print(x_train.ndim,y_train.ndim)
x_train = np.reshape(x_train,(1198,60,1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
model = Sequential()
model.add(LSTM( units = 60 ,return_sequences = True , input_shape = (60,1)))
model.add(Dropout(0.2))
model.add(LSTM( units = 60 ,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 60 , return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 60 ))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile("rmsprop",loss = "mean_squared_error")
model.fit(x_train,y_train,epochs = 1)


dataset_test = pd.read_csv('/home/riddhirup/Downloads/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2]
dataset_total = pd.concat(dataset_train["Open"],dataset_test["Open"],axis = 0) 
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60].values
inputs = inputs.reshape(-1,1)

inputs = sc.fit_transform(inputs)
x_test = []
for i in range(60,80):
    x_test.append(i-60:i,0)

x_test = np.array(x_test)
x_test = np.reshape(x_test,(20,60,1))

ypred = model.predict(x_test)
ypred = sc.invers_transform(ypred)
plt.plot(y_test,color = "red",label = "Actual Stock Price")
plt.plot(ypred,color = "blue",label = "Predicted Stock Price")
for i in range(60):
    data.append(i)
data = np.array(data)
data = np.reshape(data,(60,1))
yp = model.predict(sc.fit_transform(data))

























