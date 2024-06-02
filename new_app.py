import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import streamlit as st

# Set the title of the Streamlit app
st.title('Stock High & Low Price Prediction')

# Collect user input for stock ticker symbol
user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')

# Download stock data using yfinance
start = '2013-01-01'
end = '2024-04-30'
df = yf.download(user_input, start=start, end=end)

# Display descriptive statistics of the downloaded data
st.subheader('Data from 2013-2024')
st.write(df.describe())

# Display a chart of the high prices over time
st.subheader('High Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.High, 'g', label='High Price')
st.pyplot(fig)

# Split data into training and testing
data_training = pd.DataFrame(df['High'][0:int(len(df) * 0.75)])
data_testing = pd.DataFrame(df['High'][int(len(df) * 0.75):int(len(df))])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the trained LSTM model
model = load_model('high50.h5')

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('High Price Predictions vs Original High Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original High Price')
plt.plot(y_predicted, 'g', label='Predicted High Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predicted)
r3 = r2 * 100
r3 = str(round(r3, 2))

# Streamlit output
st.subheader('Stock High Price Prediction Model Evaluation')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')
st.write(f'R-squared (R²): {r2}')
st.write(f'Prediction Accuracy: {r3}%')

##################################################################################################################
##################################################################################################################
# st.title("Stock Low Price Prediction")

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

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predicted)
# r3 = str(round(r2*100),2)
r3 = r2 * 100
r3 = str(round(r3, 2))

# Streamlit output
st.subheader('Stock Low Price Prediction Model Evaluation')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')
st.write(f'R-squared (R²): {r2}')
st.write(f'Prediction Accuracy: {r3}%')


# Use for run this app
# http://localhost:8501/
# http://192.168.0.121:8501/
# https://kuntalojha.streamlit.app/
# streamlit run new.py
