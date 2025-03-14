from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import logging
from xgboost import XGBRegressor
import lightgbm as lgb
from requests.exceptions import Timeout
from datetime import timedelta

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Supported tickers
SUPPORTED_TICKERS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD',
    'XRP-USD', 'SOL-USD', 'DOT-USD', 'LTC-USD', 'LINK-USD',
    'MATIC-USD', 'SHIB-USD', 'AVAX-USD', 'UNI-USD', 'XLM-USD'
]
# ========== HELPER FUNCTIONS ========== #
def handle_outliers(data):
    """Handle outliers using IQR method."""
    q1 = data['Close'].quantile(0.25)
    q3 = data['Close'].quantile(0.75)
    iqr = q3 - q1
    return data[~((data['Close'] < (q1 - 1.5 * iqr)) | (data['Close'] > (q3 + 1.5 * iqr)))]

def fetch_data(ticker):
    """Fetch and preprocess data from Yahoo Finance with timeout."""
    try:
        app.logger.info(f"Fetching data for {ticker}...")
        data = yf.download(ticker, period="5y", interval='1d', auto_adjust=True, timeout=10)
        if data.empty:
            app.logger.error("No data found for this ticker")
            return None, None, "No data found for this ticker"
        
        # Flatten MultiIndex columns
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        app.logger.info(f"Original columns: {data.columns}")
        
        # Resample, handle missing values, and outliers
        data = data.resample('D').ffill()
        data = handle_outliers(data)
        
        app.logger.info(f"Data after preprocessing:\n{data.head()}")
        return data, None, None
        
    except Timeout:
        app.logger.error("Data fetch timed out")
        return None, None, "Data fetch timed out"
    except Exception as e:
        app.logger.error(f"Data fetch error: {str(e)}")
        return None, None, str(e)

def create_features(data, lags=5):
    """Create lag features and moving averages for time series forecasting."""
    app.logger.info("Creating features...")
    for i in range(1, lags + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    
    data['ma7'] = data['Close'].rolling(window=7).mean()
    data['ma30'] = data['Close'].rolling(window=30).mean()
    data['ma100'] = data['Close'].rolling(window=100).mean()
    
    data.dropna(inplace=True)
    app.logger.info(f"Data after feature engineering:\n{data.head()}")
    return data

def create_future_features(data, future_dates, lags=5):
    """Create features for future dates using the last available historical data."""
    app.logger.info("Creating features for future dates...")
    future_data = pd.DataFrame(index=future_dates)
    future_data['Close'] = np.nan  # Placeholder for Close prices
    
    # Extend lag features using the last known Close prices
    last_close = data['Close'].iloc[-1]
    for i in range(1, lags + 1):
        lag_values = data['Close'].shift(i).reindex(future_data.index, method='ffill')
        future_data[f'lag_{i}'] = lag_values
    
    # Extend moving averages (use historical data for initial windows)
    future_data['ma7'] = data['ma7'].reindex(future_data.index, method='ffill')
    future_data['ma30'] = data['ma30'].reindex(future_data.index, method='ffill')
    future_data['ma100'] = data['ma100'].reindex(future_data.index, method='ffill')
    
    return future_data

def train_xgboost(train_data):
    """Train XGBoost model with default parameters."""
    app.logger.info("Training XGBoost model with default parameters...")
    # Drop columns that won't be available for future predictions
    X = train_data.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    y = train_data['Close']
    
    app.logger.info(f"Feature matrix (X):\n{X.head()}")
    app.logger.info(f"Target variable (y):\n{y.head()}")
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    app.logger.info("XGBoost model trained successfully")
    return model

def train_lightgbm(train_data):
    """Train LightGBM model with default parameters."""
    app.logger.info("Training LightGBM model with default parameters...")
    # Drop columns that won't be available for future predictions
    X = train_data.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    y = train_data['Close']
    
    app.logger.info(f"Feature matrix (X):\n{X.head()}")
    app.logger.info(f"Target variable (y):\n{y.head()}")
    
    model = lgb.LGBMRegressor(objective='regression', n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    app.logger.info("LightGBM model trained successfully")
    return model

def calculate_metrics(forecast, data):
    """Calculate RMSE, MAE, MAPE (for historical data)."""
    merged = forecast.set_index('ds')[['yhat']].join(data['Close'])
    merged.dropna(inplace=True)
    
    close_values = merged['Close']
    mape = np.mean(np.abs((close_values - merged['yhat']) / np.where(close_values == 0, 1, close_values))) * 100
    
    return {
        'rmse': np.sqrt(mean_squared_error(merged['Close'], merged['yhat'])),
        'mae': mean_absolute_error(merged['Close'], merged['yhat']),
        'mape': mape
    }

def create_visualizations(model, forecast, historical_data, future_forecast, model_type):
    """Generate visualizations with historical and future predictions."""
    app.logger.info("Creating visualizations...")
    # Forecast plot (historical + future)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Historical'))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Historical Predicted'))
    fig1.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Future Predicted', line=dict(dash='dash')))
    fig1.update_layout(title=f'{model_type} Forecast', xaxis_title='Date', yaxis_title='Price (USD)')
    
    # Candlestick chart (historical only)
    fig2 = create_candle_chart(historical_data)
    
    # Metrics (calculated on historical forecast)
    metrics = calculate_metrics(forecast, historical_data)
    fig3 = create_metrics_chart(metrics)
    
    app.logger.info("Visualizations created successfully")
    return fig1.to_html(full_html=False), fig2.to_html(full_html=False), fig3.to_html(full_html=False), metrics

def create_candle_chart(data):
    """Create candlestick chart with moving averages."""
    ma30 = data['Close'].rolling(30).mean()
    ma100 = data['Close'].rolling(100).mean()
    fig = go.Figure(data=[
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'),
        go.Scatter(x=data.index, y=ma30, line=dict(color='orange', width=1), name='MA30'),
        go.Scatter(x=data.index, y=ma100, line=dict(color='blue', width=1), name='MA100')
    ])
    fig.update_layout(title='Historical Price Analysis', xaxis_rangeslider_visible=False)
    return fig

def create_metrics_chart(metrics):
    """Bar chart for metrics."""
    fig = go.Figure([go.Bar(x=list(metrics.keys()), y=list(metrics.values()), text=[f"{v:.2f}" for v in metrics.values()])])
    fig.update_layout(title='Model Performance Metrics', yaxis_title='Metric Value')
    return fig

# ========== ROUTES ========== #
@app.route('/')
def index():
    app.logger.info("Rendering index.html")
    return render_template('index.html', active_page='home')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Received prediction request")
    
    # Get form inputs
    ticker = request.form.get('ticker', '').strip().upper()
    days = request.form.get('days', '30').strip()
    model_type = request.form.get('model', 'xgboost')
    
    app.logger.info(f"Ticker: {ticker}, Days: {days}, Model: {model_type}")
    
    # Validate inputs
    if ticker not in SUPPORTED_TICKERS:
        app.logger.error(f"Unsupported ticker: {ticker}")
        return render_template('error.html', message="Unsupported ticker symbol"), 400
    
    if not days.isdigit() or int(days) < 1 or int(days) > 365:
        app.logger.error(f"Invalid days: {days}")
        return render_template('error.html', message="Prediction window must be between 1 and 365 days"), 400
    
    days = int(days)
    
    # Fetch data
    historical_data, _, error = fetch_data(ticker)
    if error:
        app.logger.error(f"Data fetch failed: {error}")
        return render_template('error.html', message=error), 400
    
    app.logger.info("Data fetched successfully")
    
    # Create features for historical data
    historical_data = create_features(historical_data)
    app.logger.info("Historical features created")
    
    # Train model on all historical data
    app.logger.info(f"Training {model_type} model...")
    if model_type == 'xgboost':
        model = train_xgboost(historical_data)
    elif model_type == 'lightgbm':
        model = train_lightgbm(historical_data)
    else:
        app.logger.error(f"Invalid model type: {model_type}")
        return render_template('error.html', message="Invalid model selected"), 400
    
    app.logger.info("Model trained successfully")
    
    # Generate predictions on historical data (for metrics and visualization)
    X_historical = historical_data.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    y_historical = historical_data['Close']
    y_pred_historical = model.predict(X_historical)
    forecast = pd.DataFrame({'ds': historical_data.index, 'yhat': y_pred_historical})
    forecast['trend'] = forecast['yhat'].diff().apply(lambda x: 1 if x > 0 else -1)
    
    # Generate future dates (from tomorrow onwards)
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
    app.logger.info(f"Future dates generated: {future_dates[0]} to {future_dates[-1]}")
    
    # Create features for future dates
    future_data = create_future_features(historical_data, future_dates)
    
    # Predict iteratively for future dates
    future_predictions = []
    current_data = historical_data.copy()
    for i in range(days):
        # Prepare features for the next day
        X_future = future_data.drop(columns=['Close']).iloc[[i]]
        # Predict the next day's price
        y_pred = model.predict(X_future)[0]
        future_predictions.append(y_pred)
        # Update the Close price for the next iteration
        future_data['Close'].iloc[i] = y_pred
        # Update lag features
        for lag in range(1, 6):
            if i + lag < days:
                future_data[f'lag_{lag}'].iloc[i + lag] = y_pred
        # Update moving averages
        if i + 7 < days:
            future_data['ma7'].iloc[i + 7] = future_data['Close'].iloc[max(0, i-6):i+1].mean()
        if i + 30 < days:
            future_data['ma30'].iloc[i + 30] = future_data['Close'].iloc[max(0, i-29):i+1].mean()
        if i + 100 < days:
            future_data['ma100'].iloc[i + 100] = future_data['Close'].iloc[max(0, i-99):i+1].mean()
    
    # Prepare future forecast DataFrame
    future_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })
    future_forecast['trend'] = future_forecast['yhat'].diff().apply(lambda x: 1 if x > 0 else -1)
    predictions = future_forecast[['ds', 'yhat']].values.tolist()
    
    # Prepare visualizations (historical + future)
    app.logger.info("Creating visualizations...")
    plot_html, candle_html, metrics_html, metrics = create_visualizations(model, forecast, historical_data, future_forecast, model_type)
    app.logger.info("Visualizations created")
    
    app.logger.info("Rendering results.html")
    return render_template('results.html', 
                         plot_html=plot_html, 
                         candle_html=candle_html,
                         metrics_html=metrics_html, 
                         metrics=metrics, 
                         predictions=predictions,
                         ticker=ticker, 
                         days=days, 
                         model_type=model_type)

@app.route('/eda')
def eda():
    app.logger.info("Received EDA request")
    ticker = request.args.get('ticker', 'BTC-USD')
    data, _, error = fetch_data(ticker)
    if error:
        app.logger.error(f"EDA fetch failed: {error}")
        return render_template('error.html', message=error), 400
    
    try:
        app.logger.info("Generating EDA plots...")
        price_trend = plot_price_trend(data)
        volatility = plot_volatility(data)
        correlation = plot_correlation(data)
        returns_dist = plot_returns_distribution(data)
        
        app.logger.info("Rendering eda.html")
        return render_template('eda.html', 
                             price_trend=price_trend,
                             volatility=volatility,
                             correlation=correlation,
                             returns_dist=returns_dist,
                             ticker=ticker)
    except Exception as e:
        app.logger.error(f"Error generating EDA plots: {str(e)}")
        return render_template('error.html', message="An error occurred while generating EDA plots."), 500

def plot_price_trend(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Price Trend', xaxis_title='Date', yaxis_title='Price (USD)')
    return fig.to_html(full_html=False)

def plot_volatility(data):
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(30).std()
    data.dropna(inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='30-Day Volatility'))
    fig.update_layout(title='Volatility Analysis', xaxis_title='Date', yaxis_title='Volatility')
    return fig.to_html(full_html=False)

def plot_correlation(data):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        app.logger.error("Missing required columns for correlation heatmap")
        return "<p>Error: Missing required columns for correlation heatmap.</p>"
    
    corr = data[required_columns].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap')
    return fig.to_html(full_html=False)

def plot_returns_distribution(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    fig = go.Figure(data=[go.Histogram(x=data['Returns'], nbinsx=50)])
    fig.update_layout(title='Distribution of Daily Returns', xaxis_title='Returns', yaxis_title='Frequency')
    return fig.to_html(full_html=False)

@app.route('/about')
def about():
    app.logger.info("Rendering about.html")
    return render_template('about.html', active_page='about')

@app.route('/contact')
def contact():
    app.logger.info("Rendering contact.html")
    return render_template('contact.html', active_page='contact')

@app.route('/howto')
def howto():
    app.logger.info("Rendering howto.html")
    return render_template('howto.html', active_page='howto')

if __name__ == '__main__':
    app.run()