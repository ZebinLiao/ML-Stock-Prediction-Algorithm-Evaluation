# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:11:50 2023

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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

#load data
company = "NVDA"

start = dt.datetime (2012,1,1)
end = dt.datetime.now()

train_start = dt.datetime (2012,1,1)
train_end = dt.datetime(2021,12,1)

data = pdr.get_data_yahoo(company, start, end)

data['change_in_price'] = data['Close'].diff()

data.at['2012-01-03 00:00:00-05:00','change_in_price']=0

###

n=14

up_df, down_df = data[['change_in_price']].copy(), data[['change_in_price']].copy()

# For up days, if the change is less than 0 set to 0.
up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

# For down days, if the change is greater than 0 set to 0.
down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

down_df['change_in_price'] = down_df['change_in_price'].abs()

# Calculate the Exponential Weighted Moving Average
ewma_up = up_df['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
ewma_down = down_df['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

relative_strength = ewma_up / ewma_down

relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

data['down_days'] = down_df['change_in_price']
data['up_days'] = up_df['change_in_price']
data['RSI'] = relative_strength_index

data.at['2012-01-04 00:00:00-05:00','RSI']=25.3

###

###

low_14, high_14 = data[['Low']].copy(), data[['High']].copy()

low_14 = low_14['Low'].transform(lambda x: x.rolling(window = n).min())
high_14 = high_14['High'].transform(lambda x: x.rolling(window = n).max())

# Calculate the Stochastic Oscillator.
k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))

# Add the info to the data frame.
data['low_14'] = low_14
data['high_14'] = high_14
data['k_percent'] = k_percent

r_percent = ((high_14 - data['Close']) / (high_14 - low_14)) * - 100
data['r_percent'] = r_percent

###

###

# Calculate the MACD
ema_26 = data['Close'].transform(lambda x: x.ewm(span = 26).mean())
ema_12 = data['Close'].transform(lambda x: x.ewm(span = 12).mean())
macd = ema_12 - ema_26

# Calculate the EMA
ema_9_macd = macd.ewm(span = 9).mean()

# Store the data in the data frame.
data['MACD'] = macd
data['MACD_EMA'] = ema_9_macd

###

###

# Calculate the Price Rate of Change
n = 9

# Calculate the Rate of Change in the Price, and store it in the Data Frame.
data['Price_Rate_Of_Change'] = data['Close'].transform(lambda x: x.pct_change(periods = n))

###

###

def obv(group):

    change = data['Close'].diff()
    volume = data['Volume']

    # intialize the previous OBV
    prev_obv = 0
    obv_values = []

    # calculate the On Balance Volume
    for i, j in zip(change, volume):

        if i > 0:
            current_obv = prev_obv + j
        elif i < 0:
            current_obv = prev_obv - j
        else:
            current_obv = prev_obv

        prev_obv = current_obv
        obv_values.append(current_obv)
    
    # Return a panda series.
    return pd.Series(obv_values, index = data.index)
        

# apply the function
obv_groups = data.apply(obv)['Open']

# add to data 
data['On Balance Volume'] = obv_groups

###

###

close_groups = data['Close']

# Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

# add the data to the main dataframe.
data['Prediction'] = close_groups

data.loc[data['Prediction'] == 0.0] = 1.0
data['Prediction'] = data['Prediction'].shift(-1)

###

data.at['2012-01-03 00:00:00-05:00','Prediction']= 1

data.fillna(0, inplace=True)

scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))

# Group by the `Symbol` column, then grab the `Close` column.
close_groups = data['Close']

data['value_prediction'] = close_groups.shift(-1)

# Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

data.fillna(0, inplace=True)

data_train = data.loc[:train_end,:]

x_train_data = data_train[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]
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

x_test_data = data_test[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]
y_test_data = data_test ['value_prediction']

x_test_data = scaler_x.fit_transform(x_test_data)
y_test_data = scaler_y.fit_transform(y_test_data.values.reshape(-1,1))

total_dataset = pd.concat((data[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']], data_test[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]), axis = 0)

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

regressor = LinearRegression()
regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)

predicted = scaler_y.inverse_transform(predicted)

end = time.time()
print(f'Time Taken: {end-start:.3f}')

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

classifier = SGDClassifier()

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