import sys
import os
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Crypto Price Prediction" in response.data

def test_prediction_route(client):
    """Test the prediction route with valid input."""
    response = client.post('/predict', data={
        'ticker': 'BTC-USD',  # Ensure this is a valid ticker
        'days': '30'
    })
    assert response.status_code == 200
    assert b"Price Forecast" in response.data
    
def test_invalid_ticker(client):
    """Test the prediction route with an invalid ticker."""
    response = client.post('/predict', data={
        'ticker': 'INVALID-TICKER',
        'days': '30'
    })
    assert response.status_code == 400
    assert b"Error" in response.data

def test_negative_days(client):
    """Test the prediction route with negative days."""
    response = client.post('/predict', data={
        'ticker': 'BTC-USD',
        'days': '-10'
    })
    assert response.status_code == 400
    assert b"Invalid prediction window" in response.data

def test_missing_ticker(client):
    """Test the prediction route with missing ticker."""
    response = client.post('/predict', data={
        'days': '30'
    })
    assert response.status_code == 400
    assert b"Missing required fields" in response.data

def test_missing_days(client):
    """Test the prediction route with missing days."""
    response = client.post('/predict', data={
        'ticker': 'BTC-USD'
    })
    assert response.status_code == 400
    assert b"Missing required fields" in response.data