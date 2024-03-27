#this code trains an LSTM model to predict stock prices based on historical data,
#evaluates its performance, and predicts the next day's stock price.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf

plt.style.use('fivethirtyeight')

# Enable Copy-on-Write mode in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# Define the date range
start = '2010-01-01'
end = '2024-01-10'

try:
    # Retrieve data using yfinance
    stock_data = yf.download('AAPL', start=start, end=end)

    # Display the first few rows of the DataFrame
    print(stock_data.head())
    print(stock_data.shape)  # Displaying the shape of the data

    # Creating a new dataframe with only the Close column
    data = stock_data.filter(['Close'])
    dataset = data.values  # Convert dataframe to numpy array

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Creating training data set
    train_len = math.ceil(len(dataset) * .8)  # Determine the number of rows to train on
    train_data = scaled_data[:train_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)  # Adjust epochs for better training

    # Creating test data set
    test_data = scaled_data[train_len - 60:, :]
    x_test = []
    y_test = dataset[train_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert data to numpy array
    x_test = np.array(x_test)

    # Reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate root mean squared error (RMSE) to assess model accuracy
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("Root Mean Squared Error:", rmse)

    # Plotting the data
    train = data[:train_len]
    valid = data[train_len:]
    valid['Predictions'] = predictions  # Update using direct assignment

    # Visualizing the data
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.show()

    # Displaying the valid and predicted prices
    print(valid)

    # Predicting the next day's price
    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    print("Predicted price for the next day:", predicted_price)

except Exception as e:
    print("An error occurred:", e)
