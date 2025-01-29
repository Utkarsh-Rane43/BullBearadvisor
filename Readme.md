# Stock Market Analysis Dashboard

## Overview
This project provides a **Stock Market Analysis Dashboard** where users can analyze stock data, view historical price trends, and predict future stock prices using **XGBoost**. The app integrates with **Yahoo Finance** to get real-time stock information and **NewsAPI** for the latest stock-related news, while using **XGBoost** for stock price predictions.

## Features:
- **Real-Time Stock Information**: Display stock metrics like current price, day-high, and day-low.
- **News and Sentiment Analysis**: Fetch the latest news related to the selected stock and perform sentiment analysis using **TextBlob**.
- **Stock Price History**: View historical stock price data using **Plotly** for interactive charting.
- **Price Prediction**: Predict stock prices for the next 7 days using an **XGBoost** model.

## Requirements:
- Python 3.x
- Streamlit
- yfinance
- requests
- pandas
- plotly
- textblob
- scikit-learn
- xgboost

### Installation:
1. Install the required packages:
   ```bash
   pip install streamlit yfinance requests pandas plotly textblob scikit-learn xgboost
