import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ==========================
# 1️⃣ DOWNLOAD DATA
# ==========================
stock_ticker = 'RGTI'  # Change this to any ticker
end_date = datetime.today()
start_date = end_date - timedelta(days=6*30)  # Approximate 6 months
data = yf.download(stock_ticker, start=start_date, end=end_date, interval='1d')

# ==========================
# 2️⃣ PREPROCESS DATA
# ==========================
# Scale 'Close' prices only
close_prices = data[['Close']].values
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = close_scaler.fit_transform(close_prices)

# Create X (last 30 days) and y (next day's price)
lookback_period = 30
X, y = [], []
for i in range(lookback_period, len(close_prices_scaled)):
    X.append(close_prices_scaled[i - lookback_period:i])
    y.append(close_prices_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input: (samples, timesteps, features)

# ==========================
# 3️⃣ SPLIT DATA
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_dates = data.index[lookback_period + len(X_train):]  # Dates for the test set

# ==========================
# 4️⃣ BUILD & TRAIN MODEL (FINE-TUNED)
# ==========================
# Add custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  

# LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(1)
])

# Compile with Mean Absolute Error (MAE) for better accuracy
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

# ==========================
# 5️⃣ MAKE PREDICTIONS
# ==========================
# Predict the test set
predictions = model.predict(X_test)
predictions_rescaled = close_scaler.inverse_transform(predictions)
y_test_rescaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Absolute Error (better for close prices)
mae = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"Mean Absolute Error: {mae}")

# ==========================
# 6️⃣ FUTURE PREDICTIONS
# ==========================
# Predict the next 10 days
last_30_days = close_prices_scaled[-lookback_period:].reshape(1, lookback_period, 1)
future_predictions = []

for _ in range(10):
    next_day_prediction = model.predict(last_30_days)[0, 0]
    future_predictions.append(next_day_prediction)
    last_30_days = np.append(last_30_days[:, 1:, :], np.array(next_day_prediction).reshape(1, 1, 1), axis=1)

# Rescale future predictions
future_predictions_rescaled = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=data.index[-1], periods=11, freq='B')[1:]

# ==========================
# 7️⃣ PLOT RESULTS (LAST 30 DAYS + FUTURE PREDICTIONS)
# ==========================
plt.figure(figsize=(14, 6))

# Plot last 30 days of actual Close prices
plt.plot(data.index[-lookback_period:], close_prices[-lookback_period:], label='Actual Close Price (Last 30 Days)', color='blue')

# Plot predicted prices for the test set
plt.plot(test_dates, predictions_rescaled[-len(test_dates):], label='Predicted Close Price (Test)', color='red')

# Plot future predictions (smoothed)
plt.plot(future_dates, smoothed_predictions, label='Future Predictions (Smoothed)', color='green', linestyle='dashed')

# Format the graph
plt.title(f'{stock_ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
plt.show()

# ==========================
# 8️⃣ PRINT FUTURE PREDICTIONS
# ==========================
print("\nPredicted Prices for the Next 10 Days:")
for date, price in zip(future_dates, future_predictions_rescaled):
    print(f"{date.date()}: ${price[0]:.2f}")