import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime

BASE = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1h"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]

def months_between(start: str, end: str):
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end).to_period("M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]

def build_monthly_url(symbol: str, year_month: str) -> str:
    yyyy, mm = year_month.split("-")
    fname = f"{symbol.upper()}-{INTERVAL}-{yyyy}-{mm}.zip"
    return f"{BASE}/{symbol.upper()}/{INTERVAL}/{fname}"

def fetch_zip_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"Could not download: {url} (status {r.status_code})")
    return r.content

def read_klines_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Usually one CSV file inside
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_name:
            raise ValueError("Zip contained no CSV.")
        with z.open(csv_name[0]) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLS)

    ot = df["open_time"].astype("int64")
    unit = "us" if ot.max() > 10**14 else "ms"

    ts = pd.to_datetime(ot, unit=unit, utc=True)
    out = pd.DataFrame({
        "timestamp": ts,
        "close": df["close"].astype("float64"),
    })

    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out

def build_close_wide(
    symbols,
    start_date: str,
    end_date: str,
    cache_dir: str = "./binance_cache_zips",
    allow_cache: bool = True,
) -> pd.DataFrame:

    os.makedirs(cache_dir, exist_ok=True)

    ym_list = months_between(start_date, end_date)
    frames = []

    for sym in symbols:
        for ym in ym_list:
            url = build_monthly_url(sym, ym)
            yyyy, mm = ym.split("-")
            zip_name = f"{sym.upper()}-{INTERVAL}-{yyyy}-{mm}.zip"
            zip_path = os.path.join(cache_dir, zip_name)

            try:
                if allow_cache and os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        zbytes = f.read()
                else:
                    zbytes = fetch_zip_bytes(url)
                    with open(zip_path, "wb") as f:
                        f.write(zbytes)

                df = read_klines_from_zip(zbytes)
                df["symbol"] = sym.upper()
                frames.append(df)

            except FileNotFoundError:
                continue

    if not frames:
        raise RuntimeError("No data downloaded/read. Check symbols and date range.")

    long = pd.concat(frames, ignore_index=True)

    # Trim to exact requested date range (inclusive)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    long = long[(long["timestamp"] >= start_ts) & (long["timestamp"] <= end_ts)]

    wide = (
        long.pivot(index="timestamp", columns="symbol", values="close")
            .sort_index()
    )

    full_index = pd.date_range(
        wide.index.min().floor("h"),
        wide.index.max().ceil("h"),
        freq="h",
        tz="UTC"
    )
    wide = wide.reindex(full_index)


    return wide

if __name__ == "__main__":
    symbols = [
    "BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE", "BCH", "SUI", "XAUT",
    "ADA", "LINK", "AVAX", "PAXG", "LTC", "ZEC", "TRX", "XAU", "UNI",
    "PEPE", "XAG", "ENA", "TON", "PUMP", "AAVE"
]
    universe = [s + "USDT" for s in symbols]

    close_wide = build_close_wide(
        symbols=universe,
        start_date="2024-10-01",
        end_date="2025-03-31",
        cache_dir="./binance_cache_zips",
        allow_cache=True,
    )

    print(close_wide.head())
    print(close_wide.tail())

    close_wide.to_csv("close_wide_1h.csv")

