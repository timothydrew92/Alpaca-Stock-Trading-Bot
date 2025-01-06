# Alpaca Stock Trading Bot

This project is an Alpaca stock trading bot built with Python. It uses the Alpaca Paper trading system which does not utilize actual money and can be used to test algorithms and code. The bot connects to the Alpaca API to execute trades based on market conditions.

## Beginning Code

```python
import alpaca_trade_api as tradeapi

api_key = "your_api_key"
api_secret = "your_api_secret"
base_url = "https://paper-api.alpaca.markets"  # Use paper trading API

api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Example function to check account balance
def check_balance():
    account = api.get_account()
    print(f"Cash: {account.cash}")

check_balance()
