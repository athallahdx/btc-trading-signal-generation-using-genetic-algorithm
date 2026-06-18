import pandas as pd
import os
import glob

def preprocess(filepath, freq="1min"):
    print(f"Processing {os.path.basename(filepath)}...", end=" ")

    df = pd.read_csv(filepath, parse_dates=["time"])
    df = df.drop_duplicates(subset="agg_trade_id")
    df = df.sort_values("time").reset_index(drop=True)
    df["price"]    = df["price"].astype(float)
    df["qty"]      = df["qty"].astype(float)
    df["notional"] = df["price"] * df["qty"]

    # set index FIRST, then split
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)  # ensure DatetimeIndex

    buy  = df[df["side"] == "BUY"]
    sell = df[df["side"] == "SELL"]

    bars = df["price"].resample(freq).ohlc()
    bars["volume"]         = df["qty"].resample(freq).sum()
    bars["trades"]         = df["qty"].resample(freq).count()
    bars["buy_vol"]        = buy["qty"].resample(freq).sum().fillna(0)
    bars["sell_vol"]       = sell["qty"].resample(freq).sum().fillna(0)
    bars["delta"]          = bars["buy_vol"] - bars["sell_vol"]
    bars["cvd"]            = bars["delta"].cumsum()
    bars["notional"]       = df["notional"].resample(freq).sum()
    bars["buy_notional"]   = buy["notional"].resample(freq).sum().fillna(0)
    bars["sell_notional"]  = sell["notional"].resample(freq).sum().fillna(0)

    threshold = df["notional"].quantile(0.99)
    bars["large_buy_count"]  = df[(df["notional"] > threshold) & (df["side"] == "BUY")]["notional"].resample(freq).count().fillna(0)
    bars["large_sell_count"] = df[(df["notional"] > threshold) & (df["side"] == "SELL")]["notional"].resample(freq).count().fillna(0)

    bars = bars.dropna(subset=["open"])
    print(f"{len(bars):,} bars")
    return bars


def process_all(input_dir, output_dir, freq="1min"):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}\n")

    all_bars = []

    for fpath in files:
        fname_out = os.path.splitext(os.path.basename(fpath))[0] + f"_{freq}.csv"
        fpath_out = os.path.join(output_dir, fname_out)

        if os.path.exists(fpath_out):
            print(f"  already processed, skipping {os.path.basename(fpath)}")
            bars = pd.read_csv(fpath_out, parse_dates=["time"], index_col="time")
        else:
            bars = preprocess(fpath, freq=freq)
            bars.to_csv(fpath_out)

        all_bars.append(bars)

    combined = pd.concat(all_bars).sort_index()
    combined["cvd"] = (combined["buy_vol"] - combined["sell_vol"]).cumsum()

    master_path = os.path.join(output_dir, f"ALL_{freq}.csv")
    combined.to_csv(master_path)

    print(f"\n--- Summary ---")
    print(f"Days processed : {len(files)}")
    print(f"Total bars     : {len(combined):,}")
    print(f"Date range     : {combined.index[0]} → {combined.index[-1]}")
    print(f"Master file    : {master_path}")
    print(f"\nSample:\n{combined.head(5)}")

    return combined

# 2024 july to december 

combined = process_all(
    input_dir  = "BTCUSDT-2024-7-July/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-7-July/Processed_5m",
    freq       = "5min"
)

combined = process_all(
    input_dir  = "BTCUSDT-2024-8-August/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-8-August/Processed_5m",
    freq       = "5min"
)

combined = process_all(
    input_dir  = "BTCUSDT-2024-9-September/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-9-September/Processed_5m",
    freq       = "5min"
)

combined = process_all(
    input_dir  = "BTCUSDT-2024-10-October/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-10-October/Processed_5m",
    freq       = "5min"
)

combined = process_all(
    input_dir  = "BTCUSDT-2024-11-November/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-11-November/Processed_5m",
    freq       = "5min"
)

combined = process_all(
    input_dir  = "BTCUSDT-2024-12-December/Raw_Orderflow",
    output_dir = "BTCUSDT-2024-12-December/Processed_5m",
    freq       = "5min"
)