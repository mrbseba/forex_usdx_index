import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # For fetching data from Yahoo Finance
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['Avg'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    scaled_data = scaler.fit_transform(df['Avg'].values.reshape(-1, 1))
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

# Streamlit app
def main():
    st.title("USDX Price Prediction App")
    
    model_lstm = load_model("model_lstm.h5")
    model_rnn = load_model("model_rnn_forex.h5")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a Page", ["Analysis", "Prediction"])
    
    if page == "Analysis":
        st.header("USDX Price Analysis")
        df = yf.download("DX=F", interval='1m', period='7d')
        
        st.subheader("Recent Minute-level Data")
        st.write(df.head())
        
        df['Avg'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        fig = px.line(df, x=df.index, y='Avg', labels={'index': 'Date', 'value': 'Average Price'}, title='Average USDX Price')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Prediction":
        st.header("USDX Price Predictions")
        df = yf.download("DX=F", interval='1m', period='7d')
        
        st.subheader("Recent Minute-level Data")
        st.write(df.head())
        
        data, scaler = preprocess_data(df)
        
        num_days = st.slider("Number of Days to Predict:", 1, 30, 5)
        lstm_predictions = make_predictions(model_lstm, data, scaler, num_days)
        rnn_predictions = make_predictions(model_rnn, data, scaler, num_days)
        
        last_date = df.index[-1]
        date_range = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=num_days, freq='D')
        lstm_df = pd.DataFrame(lstm_predictions, columns=['LSTM Predicted Price'], index=date_range)
        rnn_df = pd.DataFrame(rnn_predictions, columns=['RNN Predicted Price'], index=date_range)
        
        combined_df = pd.concat([df['Avg'].tail(60), lstm_df['LSTM Predicted Price'], rnn_df['RNN Predicted Price']], axis=1)
        combined_df.columns = ['Average Price', 'LSTM Predicted Price', 'RNN Predicted Price']
        
        combined_fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns, labels={'index': 'Date', 'value': 'Price'}, title='Historical and Predicted USDX Prices')
        st.plotly_chart(combined_fig, use_container_width=True)

if __name__ == "__main__":
    main()
