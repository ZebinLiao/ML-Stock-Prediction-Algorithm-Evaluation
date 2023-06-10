# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:15:18 2023

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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

#load data
company = "GOOG"

start = dt.datetime (2012,1,1)
end = dt.datetime.now()

train_start = dt.datetime (2012,1,1)
train_end = dt.datetime(2021,12,1)

data = pdr.get_data_yahoo(company, start, end)

scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))

# Group by the `Symbol` column, then grab the `Close` column.
close_groups = data['Close']

data['value_prediction'] = close_groups.shift(-1)

close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

data['Prediction'] = close_groups

data.loc[data['Prediction'] == 0.0] = 1.0
data['Prediction'] = data['Prediction'].shift(-1)

data.fillna(0, inplace=True)

data_train = data.loc[:train_end,:]

x_train_data = data_train[['Open','High','Low','Volume']]
y_train_data = data_train ['value_prediction']

x_train_data = scaler_x.fit_transform(x_train_data)
y_train_data = scaler_y.fit_transform(y_train_data.values.reshape(-1,1))

x_train = []
y_train = []

x_train = np.array(x_train_data)
y_train = np.array(y_train_data)

test_start = dt.datetime(2021,12,1)
test_end = dt.datetime.now()

data_test = data.loc[test_start:,:]

x_test_data = data_test[['Open','High','Low','Volume']]
y_test_data = data_test ['value_prediction']

x_test_data = scaler_x.fit_transform(x_test_data)
y_test_data = scaler_y.fit_transform(y_test_data.values.reshape(-1,1))

total_dataset = pd.concat((data[['Open', 'High', 'Low', 'Volume']], data_test[['Open', 'High', 'Low', 'Volume']]), axis = 0)

x_test = []
y_test = []
    
x_test = np.array(x_test_data)
y_test = np.array(y_test_data)

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

y_test = y_test[:-1]

x_test = x_test[:-1]

import time
start = time.time()

regressor = RandomForestRegressor(n_estimators=150, random_state=35)
regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)

end = time.time()
print(f'Time Taken: {end-start:.3f}')

predicted = predicted.reshape(-1,1)
predicted = scaler_y.inverse_transform(predicted)

actual_prices = data_test['value_prediction']
actual_prices = actual_prices[:-1].values


plt.plot(actual_prices, color = 'black', label = f"Actual {company} price LR")
plt.plot(predicted, color = 'green', label = f"Predicted {company} price LR")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

mse = mean_squared_error(actual_prices, predicted)

mae = mean_absolute_error(actual_prices, predicted)
r2 = r2_score(actual_prices, predicted)

rmse = math.sqrt(mse)
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')
print(f'r2: {r2:.3f}')


classifier = RandomForestClassifier(n_estimators=150, random_state=50)

y_train_data = data_train['Prediction']
y_test_data = data_test['Prediction']
y_train = np.array(y_train_data)
y_test = np.array(y_test_data)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

x_train_data = np.array(x_train_data)

y_test_data = y_test_data[:-1]

accuracy = accuracy_score(y_test_data, y_pred, normalize=(True))
print(f'Accuracy on test data: {accuracy:.3f}')