import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime

st.title("Stock Price Predictor App")

# -------------------------------
# User input
# -------------------------------
stock = st.text_input("Enter the Stock Ticker", "GOOG")

# -------------------------------
# Define date range
# -------------------------------
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# -------------------------------
# Load stock data with caching
# -------------------------------
@st.cache_data
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.dropna(inplace=True)
    return data

google_data = load_stock_data(stock, start, end)

st.subheader("Stock Data (Last 5 rows)")
st.write(google_data.tail())

# -------------------------------
# Plotting function
# -------------------------------
def plot_graph(figsize, values, full_data, extra_dataset=None, labels=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label=labels[0] if labels else None)
    plt.plot(full_data['Close'], 'b', label=labels[1] if labels else None)
    if extra_dataset is not None:
        plt.plot(extra_dataset, 'g', label=labels[2] if labels else None)
    if labels:
        plt.legend()
    return fig

# -------------------------------
# Moving averages
# -------------------------------
for window in [250, 200, 100]:
    st.subheader(f'Original Close Price and MA for {window} days')
    google_data[f'MA_{window}'] = google_data['Close'].rolling(window).mean()
    st.pyplot(plot_graph((15,6), google_data[f'MA_{window}'], google_data, labels=[f'MA_{window}', 'Close']))


# Example combined MA plot
st.subheader('Original Close Price vs MA 100 and MA 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_100'], google_data, extra_dataset=google_data['MA_250'], 
                      labels=['MA_100','Close','MA_250']))

# -------------------------------
# Split data for prediction
# -------------------------------
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Close'][splitting_len:])
x_test.columns = ['Close']  # Ensure column name

# -------------------------------
# Scale the data
# -------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# -------------------------------
# Create sequences for prediction
# -------------------------------
sequence_len = 100
x_data, y_data = [], []

for i in range(sequence_len, len(scaled_data)):
    x_data.append(scaled_data[i-sequence_len:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# -------------------------------
# Load trained model
# -------------------------------
model = load_model("Latest_stock_price_model.keras")

# -------------------------------
# Predict
# -------------------------------
predictions = model.predict(x_data)

# Inverse scaling
inv_predictions = scaler.inverse_transform(predictions)
inv_y_data = scaler.inverse_transform(y_data)

# -------------------------------
# Create DataFrame to show
# -------------------------------
plotting_data = pd.DataFrame({
    'Original Test Data': inv_y_data.reshape(-1),
    'Predictions': inv_predictions.reshape(-1)
}, index=google_data.index[splitting_len + sequence_len:])

st.subheader("Original Test Data vs Predicted Data")
st.write(plotting_data)

# -------------------------------
# Final plot: Train + Test + Predicted
# -------------------------------
st.subheader('Close Price: Train Data vs Test vs Predicted')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data['Close'][:splitting_len + sequence_len], label='Train Data')
plt.plot(plotting_data['Original Test Data'], label='Original Test Data')
plt.plot(plotting_data['Predictions'], label='Predicted Test Data')
plt.legend()
st.pyplot(fig)