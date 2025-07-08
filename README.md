# Time Series Forecasting Project

## Overview
This project implements and compares multiple time series forecasting models to predict Apple Inc. (AAPL) stock prices. The project evaluates traditional statistical models (ARIMA, SARIMA), modern forecasting tools (Facebook Prophet), and deep learning approaches (LSTM) to determine the best performing model for stock price prediction.

## Dataset
- **Source**: Apple Inc. (AAPL) stock data
- **Data Loading**: Yahoo Finance API via yfinance library
- **Features**: Open, High, Low, Close, Volume, Adjusted Close
- **Target Variable**: Close price (primary focus for forecasting)

## Models Implemented

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- Traditional statistical time series model
- Captures linear relationships and trends
- Parameters: (p, d, q) determined through ACF/PACF analysis

### 2. SARIMA (Seasonal ARIMA)
- Extension of ARIMA with seasonal components
- Parameters: (p, d, q)(P, D, Q, s) where s is seasonal period
- Better handling of seasonal patterns in stock data

### 3. Facebook Prophet
- Robust forecasting tool developed by Facebook
- Handles seasonality, holidays, and trend changes
- Easy to tune and interpret
- Good performance on business time series

### 4. LSTM (Long Short-Term Memory)
- Deep learning approach using recurrent neural networks
- Captures complex non-linear patterns
- Excellent for sequence prediction tasks
- Memory cells handle long-term dependencies

## Results Summary

| Model | RMSE | Performance |
|-------|------|-------------|
| ARIMA | Higher | Moderate |
| SARIMA | Higher | Moderate |
| Facebook Prophet | Lower | Good |
| **LSTM** | **Lowest** | **Best** |

## Key Findings

### Model Performance Analysis
- **LSTM** demonstrated superior performance with the lowest RMSE
- **Facebook Prophet** showed good results, second-best performance
- Traditional models (ARIMA, SARIMA) had higher RMSE values
- LSTM was selected as the final model due to its consistent performance

### Evaluation Criteria
The models were evaluated across three key dimensions:

1. **Short-term Forecasting**: Predictions for 1-7 days
2. **Long-term Forecasting**: Predictions for weeks to months
3. **Seasonality Handling**: Ability to capture recurring patterns

LSTM excelled in all three categories, making it the optimal choice for this stock price forecasting task.

## Installation Requirements

```bash
pip install numpy pandas matplotlib seaborn
pip install yfinance
pip install scikit-learn
pip install tensorflow keras
pip install fbprophet
pip install statsmodels
pip install pmdarima
```

## Data Loading Code

```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_aapl_data(start_date='2018-01-01', end_date=None):
    """
    Load Apple (AAPL) stock data from Yahoo Finance
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
    pandas.DataFrame: Stock data with OHLCV columns
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download AAPL data
    aapl = yf.download('AAPL', start=start_date, end=end_date)
    
    # Reset index to make Date a column
    aapl.reset_index(inplace=True)
    
    # Display basic info
    print(f"Data loaded successfully!")
    print(f"Date range: {aapl['Date'].min()} to {aapl['Date'].max()}")
    print(f"Total records: {len(aapl)}")
    print(f"Columns: {list(aapl.columns)}")
    
    return aapl

# Example usage
if __name__ == "__main__":
    # Load data
    df = load_aapl_data(start_date='2018-01-01', end_date='2024-01-01')
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display data info
    print("\nData Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Save to CSV (optional)
    df.to_csv('AAPL.csv', index=False)
    print("\nData saved to AAPL.csv")
```

## Project Structure

```
time_series_forecasting/
├── arima_model.ipynb
├── sarima_model.ipynb
├── prophet_model.ipynb
├── lstm_model.ipynb
└── README.md
```

## Usage

1. **Data Loading**: Use the provided code to load AAPL data from Yahoo Finance
2. **Model Implementation**: Run each Jupyter notebook to implement and evaluate models:
   - `arima_model.ipynb` - ARIMA model implementation
   - `sarima_model.ipynb` - SARIMA model implementation  
   - `prophet_model.ipynb` - Facebook Prophet model implementation
   - `lstm_model.ipynb` - LSTM model implementation and final selection
3. **Model Comparison**: Compare RMSE values across all models
4. **Final Model**: Use LSTM model for forecasting based on superior performance

## Key Features

- Comprehensive model comparison framework
- Robust evaluation metrics (RMSE, MAE, MAPE)
- Seasonal pattern analysis
- Short-term and long-term forecasting capabilities
- Easy-to-use data loading utilities
- Reproducible results with proper random seeds

## Future Enhancements

- Integration of additional features (volume, technical indicators)
- Ensemble methods combining multiple models
- Real-time prediction pipeline
- Web dashboard for model monitoring
- Advanced hyperparameter tuning

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- pandas, numpy, matplotlib
- yfinance for data collection
- scikit-learn for preprocessing
- statsmodels for ARIMA/SARIMA
- fbprophet for Prophet model

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## Contact
For questions or collaborations, please reach out through the project repository.
