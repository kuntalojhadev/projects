import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st 
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = '2013-01-01'
end = '2024-04-30'

st.title('Stock High & Low Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')
df = yf.download(user_input, start=start, end=end)
df.head()

# Describing Data
st.subheader('Data from 2013-2024')
st.write(df.describe())

# High Price Prediction
st.subheader("Stock High Price Prediction")

# High Price vs Time Chart
st.subheader('High Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['High'])
st.pyplot(fig)

# Splitting Data into training and testing
data_training_high = df['High'][0:int(len(df)*0.75)]
data_testing_high = df['High'][int(len(df)*0.75):]

scaler_high = MinMaxScaler(feature_range=(0, 1))
data_training_high_array = scaler_high.fit_transform(np.array(data_training_high).reshape(-1,1))

# Load high price model
model_high = load_model('high50.h5')

# Testing part for high price
past_100_days_high = data_training_high.tail(100)
final_df_high = pd.concat([past_100_days_high, data_testing_high], ignore_index=True)
input_data_high = scaler_high.transform(np.array(final_df_high).reshape(-1,1))

# Create test dataset
x_test_high = []
y_test_high = []

for i in range(100, len(input_data_high)):
    x_test_high.append(input_data_high[i-100:i])
    y_test_high.append(input_data_high[i, 0])

x_test_high, y_test_high = np.array(x_test_high), np.array(y_test_high)

# Making Predictions for high price
y_predicted_high = model_high.predict(x_test_high)
scale_factor_high = 1 / scaler_high.scale_[0]
y_predicted_high = y_predicted_high * scale_factor_high
y_test_high = y_test_high * scale_factor_high

# Final Graph for high price prediction
st.subheader('High Price Predictions vs Original High Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test_high, 'b', label='Original High Price')
plt.plot(y_predicted_high, 'r', label='Predicted High Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Low Price Prediction
st.subheader("Stock Low Price Prediction")

# Low Price vs Time Chart
st.subheader('Low Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Low'])
st.pyplot(fig)

# Splitting Data into training and testing
data_training_low = df['Low'][0:int(len(df)*0.75)]
data_testing_low = df['Low'][int(len(df)*0.75):]

scaler_low = MinMaxScaler(feature_range=(0, 1))
data_training_low_array = scaler_low.fit_transform(np.array(data_training_low).reshape(-1,1))

# Load low price model
model_low = load_model('low50.h5')

# Testing part for low price
past_100_days_low = data_training_low.tail(100)
final_df_low = pd.concat([past_100_days_low, data_testing_low], ignore_index=True)
input_data_low = scaler_low.transform(np.array(final_df_low).reshape(-1,1))

# Create test dataset
x_test_low = []
y_test_low = []

for i in range(100, len(input_data_low)):
    x_test_low.append(input_data_low[i-100:i])
    y_test_low.append(input_data_low[i, 0])

x_test_low, y_test_low = np.array(x_test_low), np.array(y_test_low)

# Making Predictions for low price
y_predicted_low = model_low.predict(x_test_low)
scale_factor_low = 1 / scaler_low.scale_[0]
y_predicted_low = y_predicted_low * scale_factor_low
y_test_low = y_test_low * scale_factor_low

# Final Graph for low price prediction
st.subheader('Low Price Predictions vs Original Low Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test_low, 'b', label='Original Low Price')
plt.plot(y_predicted_low, 'r', label='Predicted Low Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
