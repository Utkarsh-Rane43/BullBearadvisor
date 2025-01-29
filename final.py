import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load stock symbols
df = pd.read_csv('All_Stocks_Data.csv')
symbols = df['symbol'].unique().tolist()

# App
st.title('Stock Market Analysis Dashboard')

# Select a stock
selected_stock = st.selectbox('Select a Stock', symbols)
stock_symbol = selected_stock + ".NS"

# NewsAPI setup
NEWS_API_KEY = '9d01ca71d0114b77ae22e01d1d230f1f'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

try:
    # Fetch stock data
    stock = yf.Ticker(stock_symbol)
    stock_info = stock.info
    current_price = stock_info.get('currentPrice', 0)

    # Display stock information
    st.header('Stock Information')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"₹{current_price}")
    with col2:
        st.metric("Day High", f"₹{stock_info.get('dayHigh', 0)}")
    with col3:
        st.metric("Day Low", f"₹{stock_info.get('dayLow', 0)}")

    # Fetch relevant news
    st.header('Relevant News and Sentiment Analysis')
    params = {
        'q': f"{selected_stock} stock",  # Using stock symbol directly
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'relevance',
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d')
    }
    response = requests.get(NEWS_API_URL, params=params)

    sentiments = []
    if response.status_code == 200:
        news_data = response.json()
        news_items = news_data.get('articles', [])

        if news_items:
            for item in news_items[:5]:  # Display top 5 news articles
                st.subheader(item['title'])
                st.write(f"Source: {item['source']['name']}")
                st.write(f"Published At: {item['publishedAt']}")
                st.write(f"[Read More]({item['url']})")

                # Sentiment Analysis
                sentiment = TextBlob(item['description'] or "").sentiment.polarity
                sentiments.append(sentiment)

                # Categorize the sentiment
                sentiment_category = (
                    "Positive" if sentiment > 0 else
                    "Negative" if sentiment < 0 else
                    "Neutral"
                )
                st.write(f"Sentiment: {sentiment_category}")
                st.write("---")

            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            overall_sentiment = (
                "Positive" if avg_sentiment > 0 else
                "Negative" if avg_sentiment < 0 else
                "Neutral"
            )
            st.subheader("Overall Sentiment")
            st.write(f"The overall sentiment for {selected_stock} is **{overall_sentiment}**.")
        else:
            st.write("No relevant news found for this stock.")
    else:
        st.error("Error fetching news. Please try again later.")

    # Historical data and plotting
    st.header('Stock Price History')
    hist = stock.history(period='1y')
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    )])
    fig.update_layout(title=f"{selected_stock} Stock Price History")
    st.plotly_chart(fig)

    # Download historical data
    st.header('Download Data')
    csv = hist.to_csv()
    st.download_button(
        label="Download Stock Data",
        data=csv,
        file_name=f'{selected_stock}_stock_data.csv',
        mime='text/csv'
    )

    # Stock price prediction using XGBoost
    st.header('Stock Price Prediction')

    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist['Date'] = hist['Date'].map(datetime.toordinal)

    X = hist[['Date']]
    y = hist['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    accuracy = 100 - (mse / y_test.mean() * 100)

    # Predict next 7 days
    last_date = hist['Date'].iloc[-1]
    future_dates = [last_date + i for i in range(1, 8)]
    future_predictions = model.predict([[date] for date in future_dates])

    st.write(f"Model Accuracy: **{accuracy:.2f}%**")
    st.write("Note: This prediction is based on historical data. All investments are subject to Market risk and Decisions should be taken Personally.")

    prediction_df = pd.DataFrame({
        'Date': [datetime.fromordinal(date).strftime('%Y-%m-%d') for date in future_dates],
        'Predicted Price': future_predictions
    })

    st.write(prediction_df)

except Exception as e:
    st.error(f"Error: {e}")
