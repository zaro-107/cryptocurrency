# live_crypto_dashboard_full.py
# ================================================
# LIVE MULTI-CRYPTO DASHBOARD WITH ML PREDICTIONS
# - Volume-based features
# - Top 5 gainers / losers (24h)
# - Download buttons for predictions & raw data
# ================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import io

st.set_page_config(page_title="Live Crypto Dashboard", layout="wide")
st.title("Live Multi-Crypto Dashboard with ML Predictions")
st.markdown("Real-time cryptocurrency prices with next-day trend & price predictions. Features: volume-based features, top gainers/losers, and CSV downloads.")

# ---- AUTO REFRESH EVERY HOUR ----
st_autorefresh(interval=3600000, key="crypto_refresh")  # 3600000 ms = 1 hour

# ---- SYMBOL DICTIONARY ----
coins_dict = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Dogecoin (DOGE)": "dogecoin",
    "Solana (SOL)": "solana",
    "Cardano (ADA)": "cardano",
    "Litecoin (LTC)": "litecoin",
    "Binance Coin (BNB)": "binancecoin",
    "Ripple (XRP)": "ripple",
    "Polkadot (DOT)": "polkadot",
    "Avalanche (AVAX)": "avalanche-2",
    "Shiba Inu (SHIB)": "shiba-inu",
    "Tron (TRX)": "tron",
    "Chainlink (LINK)": "chainlink",
    "Polygon (MATIC)": "matic-network",
    "Stellar (XLM)": "stellar"
}

# ---- SESSION STATE SETUP ----
if "selected_coins" not in st.session_state:
    st.session_state.selected_coins = ["Bitcoin (BTC)", "Ethereum (ETH)"]

# ---- TOP GAINERS / LOSERS SECTION ----
st.markdown("## 🔥 Top 5 Gainers & ⚠️ Top 5 Losers (24h)")

@st.cache_data(ttl=300)
def fetch_markets(vs_currency="usd", per_page=50, page=1):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "price_change_percentage": "24h"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

markets = fetch_markets(per_page=100, page=1)
if markets:
    df_markets = pd.DataFrame(markets)
    # use price_change_percentage_24h (CoinGecko uses that key)
    if "price_change_percentage_24h" in df_markets.columns:
        sorted_by_change = df_markets.sort_values(by="price_change_percentage_24h", ascending=False)
        top_gainers = sorted_by_change.head(5)
        top_losers = sorted_by_change.tail(5).sort_values(by="price_change_percentage_24h")
    else:
        top_gainers = df_markets.head(5)
        top_losers = df_markets.tail(5)
else:
    top_gainers = pd.DataFrame()
    top_losers = pd.DataFrame()

col1, col2, col3 = st.columns([3, 1, 3])
with col1:
    st.subheader("Top 5 Gainers (24h)")
    if not top_gainers.empty:
        st.table(top_gainers[["name", "symbol", "current_price", "price_change_percentage_24h", "total_volume"]].rename(columns={
            "name": "Name", "symbol": "Symbol", "current_price": "Price (USD)",
            "price_change_percentage_24h": "24h %", "total_volume": "24h Volume"
        }).reset_index(drop=True))
    else:
        st.info("Market data unavailable.")

with col2:
    st.write("")  # spacer
    add_gainers = st.button("➕ Add Gainers to Selection")
    add_losers = st.button("➕ Add Losers to Selection")

    # Buttons update session state and rerun
    if add_gainers and not top_gainers.empty:
        # map names back to dictionary keys if possible
        names = []
        for idx, row in top_gainers.iterrows():
            # try to find key by matching id or name
            # prefer "Name (SYMBOL)" style if present in coins_dict
            possible_matches = [k for k, v in coins_dict.items() if (v == row.get("id")) or (row.get("name") in k) or (row.get("symbol") and row.get("symbol").upper() in k)]
            if possible_matches:
                names.append(possible_matches[0])
            else:
                # fallback: use name alone (won't be recognized by coins_dict later)
                names.append(row.get("name"))
        st.session_state.selected_coins = names
        st.experimental_rerun()

    if add_losers and not top_losers.empty:
        names = []
        for idx, row in top_losers.iterrows():
            possible_matches = [k for k, v in coins_dict.items() if (v == row.get("id")) or (row.get("name") in k) or (row.get("symbol") and row.get("symbol").upper() in k)]
            if possible_matches:
                names.append(possible_matches[0])
            else:
                names.append(row.get("name"))
        st.session_state.selected_coins = names
        st.experimental_rerun()

with col3:
    st.subheader("Top 5 Losers (24h)")
    if not top_losers.empty:
        st.table(top_losers[["name", "symbol", "current_price", "price_change_percentage_24h", "total_volume"]].rename(columns={
            "name": "Name", "symbol": "Symbol", "current_price": "Price (USD)",
            "price_change_percentage_24h": "24h %", "total_volume": "24h Volume"
        }).reset_index(drop=True))
    else:
        st.info("Market data unavailable.")

# ---- USER INPUT: MULTISELECT ----
st.markdown("---")
selected_coins = st.multiselect(
    "Select Cryptocurrencies:",
    list(coins_dict.keys()),
    default=st.session_state.selected_coins
)
timeframe = st.selectbox("Select Timeframe (days):", [1, 7, 30, 90, 180, 365], index=3)

if not selected_coins:
    st.warning("Please select at least one cryptocurrency to proceed.")
    st.stop()

# ---- FETCH HISTORICAL DATA (prices + volumes) ----
@st.cache_data(ttl=3600)
def fetch_coin_history(coin_id: str, days: int):
    """
    Returns DataFrame with Date, Price, Volume
    Uses CoinGecko /coins/{id}/market_chart endpoint which returns 'prices' and 'total_volumes'
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return pd.DataFrame(columns=["Date", "Price", "Volume"])
        # prices and volumes are lists of [timestamp, value] with same timestamps
        df_prices = pd.DataFrame(prices, columns=["timestamp", "Price"])
        df_vol = pd.DataFrame(volumes, columns=["timestamp_v", "Volume"])
        df = pd.concat([df_prices, df_vol["Volume"]], axis=1)
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["Date", "Price", "Volume"]].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["Date", "Price", "Volume"])

st.info("Fetching live historical data (prices + volumes)...")
coin_data_dict = {}
for coin_name in selected_coins:
    coin_id = coins_dict.get(coin_name)
    if coin_id is None:
        # skip unknown coins (maybe added from market list but not in dict)
        coin_data_dict[coin_name] = pd.DataFrame(columns=["Date", "Price", "Volume"])
    else:
        coin_data_dict[coin_name] = fetch_coin_history(coin_id, timeframe)
st.success("Live data fetched successfully!")

# ---- FEATURE ENGINEERING + ML ----
st.markdown("## 🔮 Predictions & Signals (with Volume features)")
results = []
all_raw_for_download = {}
all_predictions_for_download = []

for coin_name, df in coin_data_dict.items():
    raw_df = df.copy()
    all_raw_for_download[coin_name] = raw_df  # store raw for download

    # If insufficient historical data, skip ML for that coin
    if df.empty or len(df) < 5:
        results.append({
            "Coin": coin_name,
            "Predicted Price": "Insufficient data",
            "Predicted Trend": "N/A",
            "Signal": "N/A",
            "MAE": "N/A",
            "Accuracy": "N/A"
        })
        continue

    # Feature engineering
    df = df.copy().reset_index(drop=True)
    # Moving averages with min_periods so small timeframes still work
    df["MA7"] = df["Price"].rolling(window=7, min_periods=1).mean()
    df["MA14"] = df["Price"].rolling(window=14, min_periods=1).mean()
    df["Momentum"] = df["Price"] - df["Price"].shift(1)
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(window=7, min_periods=1).std().fillna(0)
    # Volume-based features
    df["Volume_Change"] = df["Volume"].pct_change().fillna(0)
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(window=7, min_periods=1).mean().replace(0, np.nan)).fillna(0)
    # Target: whether next price is up
    df["Trend"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
    df = df.dropna().reset_index(drop=True)

    # align features and labels (drop last row since it has no next-day label)
    features = ["Price", "MA7", "MA14", "Momentum", "Return", "Volatility", "Volume", "Volume_Change", "Volume_Ratio"]
    X = df[features].iloc[:-1].reset_index(drop=True)
    y_price = df["Price"].shift(-1).iloc[:-1].reset_index(drop=True)
    y_trend = df["Trend"].iloc[:-1].reset_index(drop=True)

    if len(X) < 5:
        results.append({
            "Coin": coin_name,
            "Predicted Price": "Insufficient data",
            "Predicted Trend": "Insufficient data",
            "Signal": "N/A",
            "MAE": "N/A",
            "Accuracy": "N/A"
        })
        continue

    # Time-ordered train/test split
    test_size = 0.2
    split_idx = max(1, int(len(X) * (1 - test_size)))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train_price = y_price.iloc[:split_idx]
    y_test_price = y_price.iloc[split_idx:]
    y_train_trend = y_trend.iloc[:split_idx]
    y_test_trend = y_trend.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    reg = LinearRegression()
    cls = LogisticRegression(solver="liblinear", max_iter=1000)

    try:
        reg.fit(X_train_scaled, y_train_price)
        cls.fit(X_train_scaled, y_train_trend)
    except Exception as e:
        results.append({
            "Coin": coin_name,
            "Predicted Price": "Model error",
            "Predicted Trend": "Model error",
            "Signal": "N/A",
            "MAE": "N/A",
            "Accuracy": "N/A"
        })
        continue

    # Predictions on test
    try:
        y_pred_price = reg.predict(X_test_scaled)
        y_pred_trend = cls.predict(X_test_scaled)
        mae = np.mean(np.abs(y_test_price.values - y_pred_price))
        acc = np.mean(y_test_trend.values == y_pred_trend)
    except Exception:
        mae = np.nan
        acc = np.nan

    # Next-day prediction using last available X (last row of X_test or X_train if test empty)
    try:
        latest_scaled = X_test_scaled[-1].reshape(1, -1)
    except Exception:
        latest_scaled = X_train_scaled[-1].reshape(1, -1)

    try:
        pred_price = float(reg.predict(latest_scaled)[0])
        pred_trend = int(cls.predict(latest_scaled)[0])
        signal = "Buy" if pred_trend == 1 else "Sell"
        pred_trend_str = "UP" if pred_trend == 1 else "DOWN"
        pred_price_rounded = round(pred_price, 4)
    except Exception:
        pred_price_rounded = "N/A"
        pred_trend_str = "N/A"
        signal = "N/A"

    results.append({
        "Coin": coin_name,
        "Predicted Price": pred_price_rounded,
        "Predicted Trend": pred_trend_str,
        "Signal": signal,
        "MAE": round(mae, 4) if not np.isnan(mae) else "N/A",
        "Accuracy": f"{acc*100:.2f}%" if not np.isnan(acc) else "N/A"
    })

    # store per-coin prediction row for download
    pred_row = {
        "Coin": coin_name,
        "Predicted_Price": pred_price_rounded,
        "Predicted_Trend": pred_trend_str,
        "Signal": signal,
        "MAE": round(mae, 4) if not np.isnan(mae) else None,
        "Accuracy": acc if not np.isnan(acc) else None,
        "Generated_At": datetime.utcnow().isoformat() + "Z"
    }
    all_predictions_for_download.append(pred_row)

# ---- DISPLAY PREDICTIONS ----
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# ---- DOWNLOAD BUTTONS ----
st.markdown("### ⤓ Download Data & Predictions")

# Predictions CSV
if len(all_predictions_for_download) > 0:
    preds_df = pd.DataFrame(all_predictions_for_download)
    csv = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions (CSV)",
        data=csv,
        file_name=f"crypto_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions available for download.")

# Raw data: offer a CSV per coin via a small expander list
with st.expander("Download raw time series CSVs (per coin)"):
    for coin_name, raw_df in all_raw_for_download.items():
        if raw_df is None or raw_df.empty:
            st.write(f"{coin_name}: No data")
            continue
        csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {coin_name} raw CSV",
            data=csv_bytes,
            file_name=f"{coin_name.replace(' ', '_')}_raw_{timeframe}d.csv",
            mime="text/csv"
        )

# ---- MULTI-COIN INTERACTIVE CHART ----
st.markdown("## 📈 Price Chart")
fig = go.Figure()
for coin_name in selected_coins:
    df = coin_data_dict.get(coin_name, pd.DataFrame())
    if df is None or df.empty:
        continue
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode="lines", name=coin_name))
fig.update_layout(title="Cryptocurrency Prices", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ---- BUY/SELL SUMMARY ----
st.markdown("## 💹 Buy/Sell Signals")
for res in results:
    coin = res.get("Coin", "Unknown")
    sig = res.get("Signal", "N/A")
    pred = res.get("Predicted Trend", "N/A")
    price = res.get("Predicted Price", "N/A")
    if sig == "Buy":
        st.success(f"{coin}: BUY (Predicted Trend: {pred}, Price: ${price})")
    elif sig == "Sell":
        st.error(f"{coin}: SELL (Predicted Trend: {pred}, Price: ${price})")
    else:
        st.info(f"{coin}: {sig} (Predicted Trend: {pred}, Price: {price})")

# ---- FOOTER NOTES ----
st.markdown("---")
st.markdown(
    "Notes: This dashboard uses CoinGecko public APIs (no API key). Predictions are simple ML baselines "
    "(Linear Regression and Logistic Regression) using engineered price & volume features. "
    "For better performance consider time-series models (ARIMA, Prophet, LSTM) and more data/features."
)
