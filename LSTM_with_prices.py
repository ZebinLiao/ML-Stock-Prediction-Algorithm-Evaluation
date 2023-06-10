# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 05:27:43 2023

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:22:01 2023

@author: ASUS
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from pandas_datareader import data as pdr
import datetime as dt

import yfinance as yf
yf.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import MinMaxScaler

#load data
company = "AMZN"

start = dt.datetime (2012,1,1)
end = dt.datetime.now()

train_start = dt.datetime (2012,1,1)
train_end = dt.datetime(2021,12,1)

data = pdr.get_data_yahoo(company, start, end)

data['change_in_price'] = data['Close'].diff()

#print (data.index)

data.at['2012-01-03 00:00:00-05:00','change_in_price']=0

###

# Copy the `Close` column.
close_groups = data['Close']

# Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

# add column to data set
data['Prediction'] = close_groups

data.loc[data['Prediction'] == 0.0] = 1.0
data['Prediction'] = data['Prediction'].shift(-1)

###

data.at['2012-01-03 00:00:00-05:00','Prediction']= 1

data.fillna(0, inplace=True)

data_train = data.loc[:train_end,:]

x_train_data = data_train[['Open', 'High', 'Low', 'Volume']]
y_train_data = data_train['Close'].values.reshape(-1,1)

scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))
x_train_data = scaler_x.fit_transform(x_train_data[['Open', 'High', 'Low', 'Volume']])
scaled_data = scaler_y.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 20

x_train = []
y_train = []

for x in range (prediction_days, len(x_train_data)):
    x_train.append(x_train_data[x - prediction_days: x])
    y_train.append(scaled_data[x , 0])
        
x_train = np.array(x_train)
y_train = np.array(y_train)

test_start = dt.datetime(2021,12,1)
test_end = dt.datetime.now()

data_test = data.loc[test_start:,:]

total_dataset = pd.concat((data_train[['Open', 'High', 'Low', 'Volume']], data_test[['Open', 'High', 'Low', 'Volume']]), axis = 0)

model_inputs = data[len(data) - len(data_test) - prediction_days:]

x_test_data = model_inputs[['Open', 'High', 'Low', 'Volume']]
y_test_data = model_inputs['Close'].values.reshape(-1,1)

x_test_data = scaler_x.fit_transform(x_test_data[['Open', 'High', 'Low', 'Volume']])

x_test = []
y_test = []

for x in range (prediction_days, len(x_test_data)):
    x_test.append(x_test_data[x - prediction_days: x])
    y_test.append(y_test_data[x , 0])
    
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

import time
start = time.time()

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],4)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 40, batch_size = 20)

end = time.time()
print(f'Time Taken: {end-start:.3f}')

predicted_prices = model.predict(x_test)
predicted_prices = scaler_y.inverse_transform(predicted_prices)

actual_prices = data_test['Close'].values

difference_list = []

for i in range(len(predicted_prices)):
    difference_list.append(predicted_prices[i] - actual_prices[i])
    
difference = abs(sum(difference_list)/len(difference_list))

for i in range(len(predicted_prices)):
    predicted_prices[i] = predicted_prices[i] + difference

plt.plot(actual_prices, color = 'black', label = f"Actual {company} price LR")
plt.plot(predicted_prices, color = 'green', label = f"Predicted {company} price LR")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

rmse = math.sqrt(mse)
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')
print(f'r2: {r2:.3f}')

y_train_data = data_train['Prediction'].values.reshape(-1,1)
y_test_data = model_inputs['Prediction'].values.reshape(-1,1)
y_train = np.array(y_train_data)
y_test = np.array(y_test_data)

y_train = y_train[:-20]
y_test = y_test[:-20]

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split = 0.1, verbose = 1)
y_pred = model.predict(x_test)

y_pred = list(map(lambda x: -1 if x<0 else 1, y_pred))

accuracy = accuracy_score(y_test, y_pred, normalize=(True))
print(f'Accuracy on test data: {accuracy:.3f}')