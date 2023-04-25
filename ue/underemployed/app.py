import numpy as np 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime as dt
import yfinance as yf
import tensorflow
from keras.models import load_model
import streamlit as st

# Define start and end dates for the data
start_date =  dt.date(2000, 1, 1)
end_date = st.date_input('End Date', dt.date(2023, 3, 1))

# Set up the Streamlit app and get user input for the stock ticker
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download data for the chosen stock ticker
df = yf.download(user_input, start=start_date, end=end_date)
df.index = pd.to_datetime(df.index)
# Describing data 
st.subheader('Data from {} to {}'.format(start_date, end_date))
st.write(df.describe())

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the closing price and moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma50 = df.Close.rolling(50).mean()
ma20 = df.Close.rolling(20).mean()

ax.plot(df.index, df.Close, label='Closing Price', linewidth=max(2/200, 1), alpha=0.8)
ax.plot(df.index, ma200, label='200MA', color='red', linewidth=max(2/200, 0.8), alpha=0.8)
ax.plot(df.index, ma100, label='100MA', color='purple', linewidth=max(2/100, 0.6), alpha=0.8)
ax.plot(df.index, ma50, label='50MA', color='green', linewidth=max(2/50, 0.5), alpha=0.8)



st.subheader('{} Stock Price ({})'.format(user_input, start_date.strftime('%Y') + ' to ' + end_date.strftime('%Y')))

ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()


import mplfinance as mpf
mpf.plot(df, type='candle', mav=(20,50,100,200), volume=True, figscale=1.2, datetime_format='%Y-%m-%d')


st.pyplot(fig)

#Splitting Data into training and testing

data_training = pd.DataFrame (df[ 'Close'][0: int(len(df)*0.70)]) 
data_testing = pd.DataFrame(df['Close'][ int(len (df)*0.70): int(len (df))])
print(data_training.shape)
print(data_testing.shape)


#Load my model
model=load_model('keras_model.h5')


#Testing Part

past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

a=scaler.scale_[0]
scale_factor=1/a
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Predictions vs Orginal')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'r',label='Orginal Price')
plt.plot(y_predicted,'g',label='Preditced Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

