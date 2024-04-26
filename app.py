import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import Orthogonal
import plotly.express as px
import traceback

# Function to load a model safely
def load_model_safely(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Error: {str(e)}")
        traceback.print_exc()
        return None

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['Average'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    scaled_data = scaler.fit_transform(df['Average'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, data, scaler, num_days):
    predictions = []
    for i in range(num_days):
        prediction = model.predict(data[-1].reshape(1, -1))
        predictions.append(prediction[0])
        data = np.append(data, prediction.reshape(1, -1), axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit main function
def main():
    st.title("USDX Price Prediction App")

    model_lstm = load_model_safely("model_lstm_second.h5")
    model_rnn = load_model_safely("model_rnn_forex.h5")

    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a Page", ["Analysis", "Prediction"])

    if page == "Analysis":
        st.header("USDX Price Analysis")
        df = yf.download("DX=F", interval='1m', period='7d')

        if df.empty:
            st.write("Failed to download data.")
            return

        st.write(df.head())

        fig = px.line(df, x=df.index, y=['Open', 'High', 'Low', 'Close'], labels={'index': 'Date', 'value': 'Price'}, title='Minute-Level USDX Price')
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Prediction" and model_lstm and model_rnn:
        st.header("USDX Price Predictions")

        df = yf.download("DX=F", interval='1m', period='7d')

        if df.empty:
            st.write("Failed to download data.")
            return

        st.write(df.head())

        data, scaler = preprocess_data(df)
        num_days = st.slider("Number of Days to Predict:", 1, 30, 5)

        lstm_predictions = make_predictions(model_lstm, np.copy(data), scaler, num_days)
        rnn_predictions = make_predictions(model_rnn, np.copy(data), scaler, num_days)

        last_date = df.index[-1]
        date_range = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=num_days, freq='D')

        lstm_df = pd.DataFrame(lstm_predictions, columns=['LSTM Predicted Price'], index=date_range)
        rnn_df = pd.DataFrame(rnn_predictions, columns=['RNN Predicted Price'], index=date_range)

        combined_df = pd.concat([df['Average'].tail(60), lstm_df['LSTM Predicted Price'], rnn_df['RNN Predicted Price']], axis=1)
        combined_df.columns = ['Average Price', 'LSTM Predicted Price', 'RNN Predicted Price']

        combined_fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns, labels={'index': 'Date', 'value': 'Price'}, title='Historical and Predicted USDX Prices')
        st.plotly_chart(combined_fig, use_container_width=True)

if __name__ == "__main__":
    main()
