# Portfolio Theory: Optimizing Investments for Risk and Return

Investing in financial markets requires careful consideration of risk and return. One approach to maximize returns while managing risk is portfolio theory. By diversifying investments across multiple assets, investors can potentially achieve a balance between risk and return. In this article, we will explore portfolio theory and demonstrate its practical application using Python.

## Gathering Data

To begin, we need historical stock data for our analysis. We will use the `pandas-datareader` library in Python to retrieve financial data from Yahoo Finance. Let's import the necessary libraries and define a function to load the data:

```python
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime

def Load_Data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        try:
            df = pdr.get_data_yahoo(symbol, start_date, end_date, interval='1d')
            data[symbol] = df
            print(f"Data loaded successfully for symbol: {symbol}")
        except Exception as e:
            print(f"Error loading data for symbol: {symbol}")
            print(str(e))
    return data
```

This function, `Load_Data`, takes a list of stock symbols, start date, and end date as input. It retrieves the stock data from Yahoo Finance and stores it in a dictionary, where each symbol corresponds to a DataFrame containing the historical data.

## Merging Closing Prices

Next, we want to merge the closing prices of the selected stocks into a single DataFrame. We define a function, `Merge_Closing_Price`, that extracts the 'Adj Close' column from each DataFrame and merges them based on the index:

```python
def Merge_Closing_Price(data):
    df_stocks = [df['Adj Close'] for df in data.values()]
    merged_data = pd.concat(df_stocks, axis=1)
    merged_data.columns = data.keys()
    return merged_data
```

This function, `Merge_Closing_Price`, takes the stock data dictionary as input and returns a merged DataFrame. The 'Adj Close' columns from each stock are extracted and concatenated into a single DataFrame, with the symbols as column names.

## Calculating Returns and Covariance

With the merged DataFrame, we can now calculate the returns and covariance matrix. We compute the daily returns using the `pct_change()` function from pandas. Additionally, we calculate the mean daily returns and covariance matrix:

```python
returns = df_stocks.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
```

The `pct_change()` function calculates the percentage change between consecutive values, which gives us the daily returns. The mean daily returns and covariance matrix provide insights into the average returns and the relationship between different stocks.

## Portfolio Optimization

To optimize our portfolio, we can use Monte Carlo simulation and explore different combinations of asset allocations. We randomly select portfolio weights, ensuring that they sum up to 1. For each allocation, we calculate the portfolio return and volatility (standard deviation) using the mean daily returns and covariance matrix:

```python
num_portfolios = 10000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(3)
    weights /= np.sum(weights)
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    results[2, i] = results[0, i] / results[1, i]
```

In this code snippet, we iterate over a specified number of portfolios. For each portfolio, we randomly generate weights, normalize them, and calculate the portfolio return and volatility. We store the results in a numpy array for further analysis.

## Efficient Frontier

The efficient frontier represents the set of optimal portfolios that provide the highest return for a given level of risk. We can visualize the efficient frontier using a scatter plot. Let's create a scatter plot with a color gradient based on the Sharpe Ratio:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(results_frame['Volatility'], results_frame['Return'], c=results_frame['Sharpe'], cmap='RdYlBu', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
```

In this code, we plot the volatility on the x-axis, return on the y-axis, and use the Sharpe Ratio as a color gradient. The Sharpe Ratio measures the risk-adjusted return and helps identify portfolios with optimal risk-return trade-offs. We highlight the portfolio with the maximum Sharpe Ratio and the minimum volatility for reference.

## Conclusion

Portfolio theory provides a framework for investors to optimize their investments by balancing risk and return. By diversifying across different assets, investors can potentially achieve higher returns while minimizing risk. In this article, we explored the concepts of portfolio theory and demonstrated how to apply them using Python. By analyzing historical stock data, calculating returns and covariances, and optimizing portfolios, investors can make informed decisions based on data-driven insights.



LinkedIn Readme File:

## Stock Price Prediction using Geometric Brownian Motion (GBM) and Recurrent Neural Network (RNN)

This code demonstrates how to predict stock prices using two different approaches: Geometric Brownian Motion (GBM) and Recurrent Neural Network (RNN). The code is written in Python and utilizes popular libraries such as pandas, scikit-learn, and TensorFlow.

### Geometric Brownian Motion (GBM) Model

The GBM model simulates stock price movements based on the assumption of geometric Brownian motion. It involves the following steps:

1. Load historical stock price data from Yahoo Finance using the `pandas_datareader` library.
2. Define a function to simulate the next-day price based on the GBM equation.
3. Specify the parameters for the model, such as the number of predictions, number of simulations, and window size.
4. Prepare the training data by extracting the relevant price data and applying logarithmic transformations.
5. Perform predictions by simulating future prices based on the GBM model and calculating the average.
6. Plot the predicted prices along with the actual prices.

### Recurrent Neural Network (RNN) Model

The RNN model uses a neural network architecture specifically designed to capture temporal dependencies in sequential data. The implementation follows these steps:

1. Load historical stock price data from Yahoo Finance.
2. Scale the data between 0 and 1 using the `MinMaxScaler` from scikit-learn.
3. Split the data into training and testing sets.
4. Prepare the data for the RNN model by creating sequences of input-output pairs.
5. Build an RNN model using the Keras API, consisting of LSTM layers and a dense output layer.
6. Compile and train the RNN model on the training data.
7. Make predictions on the test data and inverse scale the results.
8. Plot the predicted prices along with the actual prices.

### Evaluation and Comparison

Both models are evaluated using common performance metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). The code calculates and displays these metrics for both the GBM and RNN models in a table format.

The purpose of this code is to demonstrate the application of the GBM and RNN models for stock price prediction. You can modify and adapt the code to work with different datasets and experiment with various model configurations.

To run the code, make sure to have the required libraries installed and provide the necessary inputs such as the path to the stock data CSV file and the desired parameters for the models.

Feel free to use this code as a starting point for your own stock price prediction projects and explore further improvements and extensions to the models.

