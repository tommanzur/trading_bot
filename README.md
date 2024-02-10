# Automated Trading Strategy Using Sentiment Analysis

This repository contains the implementation of an automated trading strategy that leverages machine learning and sentiment analysis to make informed trading decisions based on market news. The strategy utilizes the FinBERT model for sentiment analysis and interacts with the Alpaca API for trading actions.

## Features

- Sentiment Analysis: Uses the FinBERT model to analyze the sentiment of market news.
- Alpaca API Integration: Trades stocks using the Alpaca API, suitable for both paper and live trading.
- Automated Trading Logic: Executes buy and sell orders based on the sentiment derived from recent news headlines.
- Backtesting Capability: Includes functionality for backtesting the strategy using historical data.


## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/automated-trading-strategy.git
cd automated-trading-strategy
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Setup

Set up your environment variables for Alpaca API:

```bash
export ALPACA_API_KEY='your_alpaca_api_key'
export ALPACA_API_SECRET='your_alpaca_api_secret'
```

## Usage

To run the trading strategy, use the following command:

```python
python main.py
``` 
For backtesting the strategy, modify the main.py script to use historical data and run the same command.

## Contributing

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make your changes.
4. Commit your changes (git commit -am 'Add some feature').
5. Push to the branch (git push origin feature/YourFeature).
6. Open a pull request.
