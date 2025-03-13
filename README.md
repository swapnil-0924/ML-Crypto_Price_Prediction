# 🚀 ML Crypto Price Prediction

A Machine Learning-based Crypto Price Prediction system using Flask, XGBoost, LightGBM, and other advanced models to forecast cryptocurrency prices with interactive visualizations.

---

## 📊 Project Overview

The **ML Crypto Price Prediction** project is a Flask-based web application that leverages advanced Machine Learning models to predict cryptocurrency prices. Users can input cryptocurrency symbols and receive detailed visualizations and predictions based on historical data.

### ✅ **Features**

- **Cryptocurrency Data Fetching**: Automatically fetches real-time data using `yfinance`.
- **ML Models**: Utilizes **XGBoost** and **LightGBM** for accurate price prediction.
- **Interactive Visualizations**: Displays dynamic charts using **Plotly**.
- **Data Preprocessing**: Normalizes and scales data with `scikit-learn`.
- **Error Metrics**: Evaluates performance using RMSE and MAE.
- **User-Friendly Interface**: Simple and responsive UI for easy access.

---

## 🧰 Tech Stack

| Technology       | Description                       |
| ---------------- | --------------------------------- |
| **Python**       | Core programming language         |
| **Flask**        | Web framework for serving the app |
| **XGBoost**      | Gradient boosting ML model        |
| **LightGBM**     | Fast, scalable gradient boosting  |
| **Plotly**       | Interactive charting for visuals  |
| **yfinance**     | Fetching cryptocurrency data      |
| **scikit-learn** | Data preprocessing and evaluation |
| **NumPy**        | Numerical computing library       |

---

## 📂 Project Structure

```
Crypto_Price_Prediction/
├── crypto_prediction/
│   ├── __pycache__/
│   ├── data/                # Store datasets
│   ├── models/              # Pretrained ML models
│   ├── static/              # Static files (CSS, JS)
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── script.js
│   └── templates/           # HTML templates
│       ├── about.html
│       ├── base.html
│       ├── contact.html
│       ├── eda.html
│       ├── error.html
│       ├── howto.html
│       ├── index.html
│       └── results.html
├── tests/                   # Unit tests
├── app.py                   # Main Flask app
├── requirements.txt         # Dependencies
├── Procfile                 # For deployment
├── render.yaml              # Render deployment config
└── README.md                # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-crypto-price-prediction.git
cd ml-crypto-price-prediction
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Access the app at **[http://localhost:5000](http://localhost:5000)**.

---

## 🌐 Deployment (Render.com)

1. Ensure `requirements.txt` and `Procfile` are present.
2. Create a `render.yaml` configuration:

```yaml
services:
  - name: crypto-price-prediction
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    plan: free
    autoDeploy: true
```

3. Push your project to GitHub:

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

4. Sign in to [Render](https://render.com) and create a **New Web Service** linked to your GitHub repo.

---

## 📈 How It Works

1. **Input**: Enter a cryptocurrency symbol (e.g., BTC, ETH).
2. **Data Fetching**: Uses `yfinance` to fetch historical data.
3. **Preprocessing**: Applies Min-Max scaling.
4. **Prediction**: Models predict future prices using `XGBoost` and `LightGBM`.
5. **Visualization**: Displays charts and error metrics.

---

## 📊 Example Output

- Predicted vs. Actual Prices
- RMSE & MAE Metrics
- Interactive Line Chart of Predictions

---

## 🧪 Testing

1. Run unit tests:

```bash
pytest tests/
```

2. Ensure all tests pass before deployment.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/new-feature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push the branch: `git push origin feature/new-feature`.
5. Submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For questions or support, open an issue or contact **[swapnilajmera2399@gmail.com](mailto\:swapnilajmera2399@gmail.com)**.

