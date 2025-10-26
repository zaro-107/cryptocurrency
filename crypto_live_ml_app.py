# ================================================
# LIVE MULTI-CRYPTO DASHBOARD WITH ML PREDICTIONS
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

# ---- AUTO REFRESH EVERY HOUR ----
st_autorefresh(interval=3600000, key="crypto_refresh")  # 3600000 ms = 1 hour

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Live Crypto Dashboard", layout="wide")
st.title("Live Multi-Crypto Dashboard with ML Predictions")
st.markdown("Real-time cryptocurrency prices with next-day trend & price predictions.")

# ---- STEP 1: USER INPUT ----
coins_dict = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Dogecoin (DOGE)": "dogecoin",
    "Solana (SOL)": "solana",
    "Cardano (ADA)": "cardano",
    "Litecoin (LTC)": "litecoin"
}

selected_coins = st.multiselect("Select Cryptocurrencies:", list(coins_dict.keys()), default=["Bitcoin (BTC)", "Ethereum (ETH)"])
timeframe = st.selectbox("Select Timeframe (days):", [1, 7, 30, 90, 180, 365], index=3)

# ---- STEP 2: FETCH LIVE DATA ----
@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_coin_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'Price'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['Date', 'Price']]
    return df

st.info(f"Fetching live data for selected coins...")
coin_data_dict = {}
for coin_name in selected_coins:
    coin_id = coins_dict[coin_name]
    coin_data_dict[coin_name] = fetch_coin_data(coin_id, timeframe)
st.success(" Live data fetched successfully!")

# ---- STEP 3: FEATURE ENGINEERING + ML ----
results = []

for coin_name, df in coin_data_dict.items():
    df['MA7'] = df['Price'].rolling(7).mean()
    df['MA14'] = df['Price'].rolling(14).mean()
    df['Momentum'] = df['Price'] - df['Price'].shift(1)
    df['Trend'] = (df['Price'].shift(-1) > df['Price']).astype(int)
    df = df.dropna()

    features = ['Price', 'MA7', 'MA14', 'Momentum']
    X = df[features]
    y_price = df['Price'].shift(-1).dropna()
    y_trend = df['Trend'][:-1]
    X = X.iloc[:-1]

    # Train/Test split
    X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, shuffle=False)
    _, _, y_train_trend, y_test_trend = train_test_split(X, y_trend, test_size=0.2, shuffle=False)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    reg = LinearRegression()
    cls = LogisticRegression()

    reg.fit(X_train_scaled, y_train_price)
    cls.fit(X_train_scaled, y_train_trend)

    y_pred_price = reg.predict(X_test_scaled)
    y_pred_trend = cls.predict(X_test_scaled)

    mae = np.mean(np.abs(y_test_price - y_pred_price))
    acc = np.mean(y_test_trend == y_pred_trend)

    # Next-day prediction
    latest_scaled = X_test_scaled[-1].reshape(1, -1)
    pred_price = reg.predict(latest_scaled)[0]
    pred_trend = cls.predict(latest_scaled)[0]
    signal = "Buy" if pred_trend == 1 else "Sell"

    results.append({
        "Coin": coin_name,
        "Predicted Price": round(pred_price, 2),
        "Predicted Trend": "UP" if pred_trend == 1 else "DOWN",
        "Signal": signal,
        "MAE": round(mae, 2),
        "Accuracy": f"{acc*100:.2f}%"
    })

# ---- STEP 4: DISPLAY RESULTS ----
st.markdown("## Predictions & Signals")
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# ---- STEP 5: MULTI-COIN INTERACTIVE CHART ----
st.markdown("## 📈 Price Chart")
fig = go.Figure()
for coin_name in selected_coins:
    df = coin_data_dict[coin_name]
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name=coin_name))
fig.update_layout(title="Cryptocurrency Prices", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ---- STEP 6: BUY/SELL SIGNALS ----
st.markdown("## 💹 Buy/Sell Signals")
for res in results:
    if res["Signal"] == "Buy":
        st.success(f"{res['Coin']}: BUY (Predicted Trend: {res['Predicted Trend']}, Price: ${res['Predicted Price']})")
    else:
        st.error(f"{res['Coin']}: SELL (Predicted Trend: {res['Predicted Trend']}, Price: ${res['Predicted Price']})")

