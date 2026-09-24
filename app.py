import streamlit as st
import torch
import torch.nn as nn
import joblib
import yfinance as yf
import pandas as pd

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="NIFTY 50 Deep Learning Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# LSTM Model
# -----------------------------
class NiftyLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64):
        super(NiftyLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model():

    model = NiftyLSTM()

    model.load_state_dict(
        torch.load(
            "nifty50_lstm_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    scaler = joblib.load("nifty50_scaler.pkl")

    return model, scaler


model, scaler = load_model()


# -----------------------------
# Get NIFTY 50 data
# -----------------------------
@st.cache_data
def get_nifty_data():

    data = yf.download(
        "^NSEI",
        period="3mo",
        interval="1d",
        auto_adjust=False
    )

    data = data[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ].copy()

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()

    data = data.dropna()

    return data


live_nifty = get_nifty_data()


# -----------------------------
# Prepare features
# -----------------------------
features = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'MA20',
    'MA50',
    'Return'
]

scaled_data = scaler.transform(
    live_nifty[features]
)

last_10_days = torch.tensor(
    scaled_data[-10:],
    dtype=torch.float32
).unsqueeze(0)


# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    output = model(last_10_days)

    probability = torch.sigmoid(output).item()


prediction = "UP" if probability >= 0.5 else "DOWN"

latest_close = live_nifty['Close'].iloc[-1].item()


# -----------------------------
# Dashboard
# -----------------------------
st.title("📈 NIFTY 50 Deep Learning Dashboard")

st.write(
    "NIFTY 50 trend prediction using PyTorch LSTM"
)

st.divider()


# -----------------------------
# Main metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest NIFTY 50 Close",
        f"{latest_close:.2f}"
    )

with col2:
    st.metric(
        "UP Probability",
        f"{probability * 100:.2f}%"
    )

with col3:
    st.metric(
        "Prediction",
        prediction
    )


st.divider()


# -----------------------------
# Model results
# -----------------------------
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Accuracy",
        "54.79%"
    )

with col2:
    st.metric(
        "Test Accuracy",
        "55.17%"
    )


# -----------------------------
# Backtesting
# -----------------------------
st.subheader("Backtesting Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Strategy Return",
        "37.16%"
    )

with col2:
    st.metric(
        "Buy & Hold Return",
        "32.68%"
    )


# -----------------------------
# NIFTY 50 Price Chart
# -----------------------------
st.subheader("NIFTY 50 Price Chart")

chart_data = live_nifty['Close'].copy()

if isinstance(chart_data, pd.DataFrame):
    chart_data = chart_data.iloc[:, 0]

st.line_chart(chart_data)


# -----------------------------
# Recent Data
# -----------------------------
st.subheader("Recent NIFTY 50 Data")

st.dataframe(
    live_nifty.tail(10),
    use_container_width=True
)


# -----------------------------
# Disclaimer
# -----------------------------
st.divider()

st.caption(
    "This dashboard is developed for educational purposes. "
    "Model predictions are not financial advice or guaranteed "
    "future market movements."
)
