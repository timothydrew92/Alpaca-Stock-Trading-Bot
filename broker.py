"""Connects to Alpaca using keys from .env.paper or .env.live. Keys never live in code."""

from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent


def load_keys(mode: str) -> tuple[str, str]:
    env_file = ROOT / f".env.{mode}"
    values = dotenv_values(env_file)
    key = (values.get("APCA_API_KEY_ID") or "").strip()
    secret = (values.get("APCA_API_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise SystemExit(f"Missing keys: fill in APCA_API_KEY_ID and APCA_API_SECRET_KEY in {env_file.name}")
    return key, secret


def trading_client(mode: str) -> TradingClient:
    key, secret = load_keys(mode)
    # paper=True/False picks the endpoint, so a paper key can never reach the live API by accident.
    return TradingClient(key, secret, paper=(mode == "paper"))


def data_client(mode: str = "paper") -> StockHistoricalDataClient:
    key, secret = load_keys(mode)
    return StockHistoricalDataClient(key, secret)
