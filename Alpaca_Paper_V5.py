import os
import time
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Your Alpaca API credentials
API_KEY = "PK14VXEVLLWLDV6HJDW5"
API_SECRET = "JRx69m1C9GNTfqUYxVp9150dX8P6zhNDasQ6WfcS"
BASE_URL = "https://paper-api.alpaca.markets"  # Use for paper trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Stock parameters
STOCKS = {
    "IONQ": {"quantity": 10},
    "RGTI": {"quantity": 10},
    "NVDA": {"quantity": 10},
    "LUNR": {"quantity": 10},
    "RKLB": {"quantity": 10},
    "SMR": {"quantity": 10},
    "SMCI": {"quantity": 10},
    "SPCE": {"quantity": 10},
    "ASTS": {"quantity": 10},
    "TTD": {"quantity": 10},
}

# Logging function
def log_status(message):
    status_file = "/Users/timothydrew92/Documents/AlpacaTrading/bot_status.log"  # Update the path if needed
    try:
        with open(status_file, "a") as f:
            f.write(f"{datetime.now()}: {message}\n")
    except Exception as e:
        print(f"Error logging status: {e}")

# Fetch the latest available data for a stock
def make_api_request(symbol, timeframe='day', limit=100):
    try:
        bars = api.get_barset(symbol, timeframe, limit=limit)
        return bars[symbol]
    except Exception as e:
        log_status(f"Error fetching data for {symbol}: {e}")
        return None

# Function to update stock prices
def update_stock_prices(stocks, model):
    for symbol in stocks:
        bars = make_api_request(symbol)
        if bars:
            data = preprocess_data(bars)
            X = data[:-1]
            y = data[1:, 3]  # Predicting the closing price
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Train the model (you should ideally train it separately and load the weights)
            train_model(model, X, y)
            
            # Make prediction
            prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))
            buy_price = round(prediction[0][0] * 0.99, 2)
            sell_price = round(prediction[0][0] * 1.01, 2)
            stocks[symbol]['buy_price'] = buy_price
            stocks[symbol]['sell_price'] = sell_price
        else:
            log_status(f"No data returned for symbol: {symbol}")

# Preprocess data
def preprocess_data(bars):
    data = []
    for bar in bars:
        data.append([bar.o, bar.h, bar.l, bar.c])
    return np.array(data)

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Guardrails

# Function to check settled funds
def get_settled_cash():
    try:
        account = api.get_account()
        return float(account.cash)  # Only settled cash
    except Exception as e:
        log_status(f"Error fetching settled cash: {e}")
        return 0.0

# Function to count day trades in the past 5 days
def count_day_trades():
    try:
        activities = api.get_activities(activity_types='FILL')  # Fetch order fills
        now = datetime.now()
        five_days_ago = now - timedelta(days=5)
        day_trades = 0
        trades_by_day = {}

        for activity in activities:
            trade_time = datetime.strptime(activity.transaction_time, '%Y-%m-%dT%H:%M:%SZ')
            if trade_time >= five_days_ago:
                if activity.symbol not in trades_by_day:
                    trades_by_day[activity.symbol] = []
                trades_by_day[activity.symbol].append(trade_time.date())

        for symbol, dates in trades_by_day.items():
            if len(set(dates)) >= 2:  # Buying and selling on the same day
                day_trades += 1

        return day_trades
    except Exception as e:
        log_status(f"Error counting day trades: {e}")
        return 0

# Function to check if a position is fully settled
def is_position_settled(symbol):
    try:
        activities = api.get_activities(activity_types='FILL')
        for activity in activities:
            if activity.symbol == symbol and activity.side == 'buy':
                settlement_date = datetime.strptime(activity.settlement_date, '%Y-%m-%d')
                if settlement_date > datetime.now():
                    return False
        return True
    except Exception as e:
        log_status(f"Error checking settlement for {symbol}: {e}")
        return False

# Check if a buy trade would cause a cash liquidation
def check_cash_liquidation(trade_cost):
    settled_cash = get_settled_cash()
    if trade_cost > settled_cash:
        log_status(f"Cash liquidation violation risk. Trade cost: {trade_cost}, Settled Cash: {settled_cash}")
        return False
    return True

# Function to check if the market is open
def is_market_open():
    clock = api.get_clock()
    return clock.is_open

# Function to check current positions
def get_current_positions():
    positions = api.list_positions()
    return {position.symbol: int(position.qty) for position in positions}

# Function to get the current price of a stock
def get_current_price(symbol):
    try:
        bar = api.get_latest_trade(symbol)
        return bar.p
    except Exception as e:
        log_status(f"Error fetching current price for {symbol}: {e}")
        return None

# Function to execute trades with guardrails
def execute_trades(stocks):
    settled_cash = get_settled_cash()
    current_positions = get_current_positions()
    day_trades = count_day_trades()

    if day_trades >= 3:
        log_status(f"PDT limit reached: {day_trades} day trades in the last 5 days. Skipping trades.")
        return

    for symbol, params in stocks.items():
        quantity = params["quantity"]
        current_price = get_current_price(symbol)
        if current_price is None:
            continue

        if symbol in current_positions:
            if current_positions[symbol] >= quantity and current_price >= params['sell_price']:
                if is_position_settled(symbol):
                    log_status(f"Position settled for {symbol}. Placing sell order.")
                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side='sell',
                            type='market',
                            time_in_force='gtc'
                        )
                    except Exception as e:
                        log_status(f"Error placing sell order for {symbol}: {e}")
                else:
                    log_status(f"Position for {symbol} is not settled. Skipping sell order.")
        else:
            trade_cost = quantity * current_price
            if trade_cost <= settled_cash and check_cash_liquidation(trade_cost):
                log_status(f"Placing buy order for {symbol}. Cost: {trade_cost}, Settled Cash: {settled_cash}")
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    settled_cash -= trade_cost
                except Exception as e:
                    log_status(f"Error placing buy order for {symbol}: {e}")
            else:
                log_status(f"Insufficient settled funds or cash liquidation risk for {symbol}. Skipping buy order.")

# Main script execution
model = build_model((100, 4))  # Example input shape, adjust as needed
while True:
    try:
        if is_market_open():
            log_status("Market is open. Proceeding with updates.")
            update_stock_prices(STOCKS, model)
            execute_trades(STOCKS)
        else:
            log_status("Market is closed. Skipping updates.")
        print(f"Script ran successfully at {datetime.now()}")
    except Exception as e:
        log_status(f"Error during execution: {e}")
    time.sleep(30)