import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go
from datetime import datetime

# Function to load the model safely
def load_model_safely(model_path):
    try:
        from tensorflow.keras.layers import SimpleRNN
        custom_objects = {'Orthogonal': Orthogonal, 'SimpleRNN': SimpleRNN}
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Error: {str(e)}")
        return None

# Function to fetch data from Yahoo Finance
def get_data(symbol, interval='1m', period='7d'):
    return yf.download(symbol, interval=interval, period=period)

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Average'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    scaled_data = scaler.fit_transform(data['Average'].values.reshape(-1, 1))
    return scaled_data, scaler

# Main application function
def main():
    st.title("USDX Price Forecasting")

    model_lstm = load_model_safely("model_lstm_second.h5")
    model_rnn = load_model_safely("model_rnn_forex.h5")

    symbol = st.text_input("Enter a symbol (e.g., 'DX=F' for US Dollar Index):", value="DX=F")

    if symbol:
        data = get_data(symbol)
        if data is None or data.empty:
            st.error("Failed to retrieve data.")
        else:
            st.subheader(f"Data for {symbol}")
            st.dataframe(data.tail())

            # Plotting historical data
            plot_raw_data(data)

            if st.button("Predict"):
                if model_lstm and model_rnn:
                    perform_prediction(data, model_lstm, model_rnn)
                else:
                    st.error("Model loading failed. Cannot perform predictions.")

# Plotting function for raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
    fig.update_layout(title_text='Historical USDX Prices', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to perform predictions
def perform_prediction(data, model_lstm, model_rnn):
    data_processed, scaler = preprocess_data(data)
    predictions_lstm = model_predict(model_lstm, data_processed, scaler)
    predictions_rnn = model_predict(model_rnn, data_processed, scaler)
    plot_predictions(predictions_lstm, predictions_rnn)

# Function for model prediction
def model_predict(model, data, scaler):
    num_days = 5  # Number of days to predict
    predictions = []
    for _ in range(num_days):
        last_batch = data[-1].reshape(1, -1)
        prediction = model.predict(last_batch)
        predictions.append(prediction[0])
        data = np.append(data, prediction.reshape(1, -1), axis=0)
    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_scaled.flatten()

# Function to plot predictions
def plot_predictions(predictions_lstm, predictions_rnn):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=predictions_lstm, name='LSTM Predictions', mode='lines+markers'))
    fig.add_trace(go.Scatter(y=predictions_rnn, name='RNN Predictions', mode='lines+markers'))
    fig.update_layout(title='Predicted USDX Prices', xaxis_title='Days', yaxis_title='Price')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
