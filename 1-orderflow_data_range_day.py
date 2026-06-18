import requests
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import zipfile
import time
import os
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_binance_vision(symbol, date):
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"
    r = requests.get(url, verify=False)

    if r.status_code != 200:
        print(f"  skipped {date} (status {r.status_code})")
        return None

    with zipfile.ZipFile(BytesIO(r.content)) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, header=None, names=[
                "agg_trade_id", "price", "qty", "first_trade_id",
                "last_trade_id", "timestamp", "is_seller_maker", "is_best_match"
            ])

    if not df.empty and df["timestamp"].iloc[0] > 1e14:
        df["time"] = pd.to_datetime(df["timestamp"], unit="us")
    else:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["side"] = df["is_seller_maker"].map({True: "SELL", False: "BUY"})
    df = df[["time", "side", "price", "qty", "agg_trade_id"]]
    return df

def download_range(symbol, start_date, end_date, output_dir=None):
    if output_dir is None:
        output_dir = f"{symbol}_{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        fname    = os.path.join(output_dir, f"{symbol}-{date_str}.csv")

        # skip if already downloaded
        if os.path.exists(fname):
            print(f"  already exists, skipping {date_str}")
            current += timedelta(days=1)
            continue

        print(f"Fetching {date_str}...", end=" ")
        df = download_binance_vision(symbol, date_str)

        if df is not None:
            df.to_csv(fname, index=False)
            print(f"{len(df):,} rows saved")

        current += timedelta(days=1)
        time.sleep(0.3)

    print(f"\nDone. Files saved to: {output_dir}/")

# --- set your range here ---
# From July to December 2024
download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-07-01",
    end_date   = "2024-07-31",
    output_dir="BTCUSDT-2024-7-July/Raw_Orderflow"
)

download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-08-01",
    end_date   = "2024-08-31",
    output_dir="BTCUSDT-2024-8-August/Raw_Orderflow"
)

download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-09-01",
    end_date   = "2024-09-30",
    output_dir="BTCUSDT-2024-9-September/Raw_Orderflow"
)

download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-10-01",
    end_date   = "2024-10-31",
    output_dir="BTCUSDT-2024-10-October/Raw_Orderflow"
)

download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-11-01",
    end_date   = "2024-11-30",
    output_dir="BTCUSDT-2024-11-November/Raw_Orderflow"
)

download_range(
    symbol     = "BTCUSDT",
    start_date = "2024-12-01",
    end_date   = "2024-12-31",
    output_dir="BTCUSDT-2024-12-December/Raw_Orderflow"
)