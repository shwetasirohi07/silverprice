import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Silver Price Predictor", layout="wide")
st.title('Silver Price Prediction Dashboard')
st.markdown("Extracting the last five years of Silver (SI=F) price data from Yahoo Finance and predicting the next year's price.")

ticker = 'SI=F' # Silver futures on Yahoo Finance
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5*365)

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

with st.spinner('Downloading 5 years of silver price data...'):
    try:
        data = load_data(ticker, start_date, end_date)
        if data.empty:
            st.error('Failed to download data.')
            st.stop()
            
        # Parse close price
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data['Close'].columns:
                close_series = data['Close'][ticker]
            else:
                close_series = data['Close'].iloc[:, 0]
        else:
            close_series = data['Close']
            
        df = pd.DataFrame({'Date': data.index, 'Close': close_series.values})
        df = df.dropna()
        
        st.subheader('Last 5 Years Silver Price History')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close Price'))
        fig.update_layout(title="Historical Silver Prices", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Modeling
        st.write("---")
        st.subheader("Predicting the Next Year")
        st.markdown("We use a SARIMAX time series model evaluated using RMSE. Smaller RMSE values denote a better fit.")
        
        with st.spinner("Training the time series model..."):
            train_size = int(len(df) * 0.8)
            train, test = df.iloc[:train_size], df.iloc[train_size:]
            
            # Simple ARIMA model for demonstration
            # using order=(5, 1, 0)
            model = SARIMAX(train['Close'], order=(5, 1, 0))
            model_fit = model.fit(disp=False)
            
            predictions = model_fit.predict(start=train_size, end=len(df)-1, dynamic=False)
            rmse = np.sqrt(mean_squared_error(test['Close'], predictions))
            
            st.success(f"Model successfully trained! Tested RMSE: **{rmse:.2f}**")
            
            st.markdown("### Next 1 Year Predict (252 Trading Days)")
            # Fit on full data
            full_model = SARIMAX(df['Close'], order=(5, 1, 0))
            full_model_fit = full_model.fit(disp=False)
            
            forecast_steps = 252 # trading days
            forecast = full_model_fit.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            
            future_dates = pd.bdate_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
            fig2.add_trace(go.Scatter(x=future_dates, y=forecast_mean, mode='lines', name='Forecasted (Next 1 Year)', line=dict(color='orange')))
            fig2.update_layout(title="Silver Price: History + 1 Year Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
