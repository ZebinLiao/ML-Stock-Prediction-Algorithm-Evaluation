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

close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

# add the data to the main dataframe.
data['Prediction'] = close_groups

data.loc[data['Prediction'] == 0.0] = 1.0
data['Prediction'] = data['Prediction'].shift(-1)

###

data.at['2012-01-03 00:00:00-05:00','Prediction']= 1

data.fillna(0, inplace=True)

data_train = data.loc[:train_end,:]

x_train_data = data_train[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]
y_train_data = data_train['Close'].values.reshape(-1,1)

scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))
x_train_data = scaler_x.fit_transform(x_train_data[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']])
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

total_dataset = pd.concat((data_train[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']], data_test[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]), axis = 0)

model_inputs = data[len(data) - len(data_test) - prediction_days:]

x_test_data = model_inputs[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']]
y_test_data = model_inputs['Close'].values.reshape(-1,1)

x_test_data = scaler_x.fit_transform(x_test_data[['RSI','k_percent','r_percent','MACD','On Balance Volume','Price_Rate_Of_Change']])

x_test = []
y_test = []

for x in range (prediction_days, len(x_test_data)):
    x_test.append(x_test_data[x - prediction_days: x])
    y_test.append(y_test_data[x , 0])
    
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

#time the execution time for the model
import time
start = time.time()

#build the model
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],6)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 40, batch_size = 20)

predicted_prices = model.predict(x_test)
predicted_prices = scaler_y.inverse_transform(predicted_prices)

end = time.time()
print(f'Time Taken: {end-start:.3f}')

actual_prices = data_test['Close'].values

difference_list = []


for i in range(len(predicted_prices)):
    difference_list.append(predicted_prices[i] - actual_prices[i])
    
difference = abs(sum(difference_list)/len(difference_list))

for i in range(len(predicted_prices)):
    predicted_prices[i] = predicted_prices[i] + difference

# plot the graph
plt.plot(actual_prices, color = 'black', label = f"Actual {company} price LR")
plt.plot(predicted_prices, color = 'green', label = f"Predicted {company} price LR")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

#calculate the evaluation of the results
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

#make prediction for trend prediction
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split = 0.1, verbose = 1)
y_pred = model.predict(x_test)

y_pred = list(map(lambda x: -1 if x<0 else 1, y_pred))

accuracy = accuracy_score(y_test, y_pred, normalize=(True))
print(f'Trend prediction accuracy: {accuracy:.3f}')