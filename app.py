import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st 
from keras.models import load_model


start = '2013-01-01'
end = '2024-04-30'

st.title('Stock High & Low Price Prediction')

user_input = st.text_input('Enter Stock Ticker','SBIN.NS')
df = yf.download(user_input, start=start, end=end)
df.head()

#Describing Data
st.subheader('Data from 2013-2024')
st.write(df.describe())

st.subheader("Stock High Price Prediction")

#Visualizations
st.subheader('High Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.High,'g',label = 'High Price')
st.pyplot(fig)


## Spliting Data into training and testing

data_tarining = pd.DataFrame(df['High'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['High'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1))

data_tarining_array = scaler.fit_transform(data_tarining)

# Load my model
# Use high50.h5 for more currect
model = load_model('high50.h5')

# Testing part
# Extract the last 100 days of data
past_100_days = data_tarining.tail(100)
# Concatenate past_100_days and data_training
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# test part
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test,y_test = np.array(x_test),np.array(y_test)
# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('High Price Predictions vs Original High Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original High Price')
plt.plot(y_predicted,'g',label='Predicted High Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig2)

##################################################################################################################
##################################################################################################################
st.subheader("Stock Low Price Prediction")

#Visualizations
st.subheader('Low Price vs Time Chart')
fig3 = plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.Low,'r',label = 'Low Price')
st.pyplot(fig3)


## Spliting Data into training and testing

data_tarining = pd.DataFrame(df['Low'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Low'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1))

data_tarining_array = scaler.fit_transform(data_tarining)

# Load my model
# Use high50.h5 for more currect
model = load_model('low50.h5')

# Testing part
# Extract the last 100 days of data
past_100_days = data_tarining.tail(100)
# Concatenate past_100_days and data_training
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# test part
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test,y_test = np.array(x_test),np.array(y_test)
# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Low Price Predictions vs Original Low Price')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Low Price')
plt.plot(y_predicted,'r',label='Predicted Low Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig4)
# Use for run this app
# http://localhost:8501/
# http://192.168.0.121:8501/
# streamlit run app.py