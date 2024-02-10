import torch
from typing import Tuple
from datetime import datetime
from datetime import timedelta
from transformers import (AutoTokenizer,
AutoModelForSequenceClassification)
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api import REST 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
sentiments = ["positive", "negative", "neutral"]

# Alpaca API Configuration
API_KEY = "YOUR API KEY" 
API_SECRET = "YOUR API SECRET" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CONFIG = {
    "API_KEY": API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}


# Sentiment Analysis Model
class SentimentAnalyzer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.sentiments = ["positive", "negative", "neutral"]

    def analyze(self, news_text: str) -> Tuple[float, str]:
        tokens = self.tokenizer(news_text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            result = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)].item()
        sentiment_label = self.sentiments[torch.argmax(result)]
        return probability, sentiment_label


# Alpaca API Wrapper
class AlpacaAPI:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = REST(base_url=base_url, key_id=api_key, secret_key=api_secret)

    def get_news_headlines(self, symbol: str, start_date: str, end_date: str) -> list:
        try:
            news = self.api.get_news(symbol=symbol, start=start_date, end=end_date)
            return [n.__dict__["_raw"]["headline"] for n in news]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
        

class AutomatedTradingStrategy(Strategy):
    """
    A trading strategy that uses machine learning and sentiment analysis
    to make trading decisions based on market news.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.alpaca_api = AlpacaAPI(API_KEY, API_SECRET, BASE_URL)
        self.trading_symbol = kwargs.get("trading_symbol", "SPY")
        self.investment_fraction = kwargs.get("investment_fraction", 0.5)
        self.previous_trade = None

    def calculate_position_size(self):
        """
        Calculates the size of the trading position.
        
        Returns:
            Tuple[float, float, int]: Available cash, last price, and quantity to trade.
        """
        try:
            available_cash = self.get_cash() 
            current_price = self.get_last_price(self.trading_symbol)
            trade_quantity = round(available_cash * self.investment_fraction / current_price, 0)
            return available_cash, current_price, trade_quantity
        except Exception as e:
            print(f"Error in calculate_position_size: {e}")
            return 0, 0, 0

    def retrieve_trade_dates(self):
        """
        Retrieves the current and past dates for news retrieval.

        Returns:
            Tuple[str, str]: Current date and date three days prior in 'YYYY-MM-DD' format.
        """
        try:
            current_date = self.get_datetime()
            date_three_days_ago = current_date - timedelta(days=3)
            return current_date.strftime('%Y-%m-%d'), date_three_days_ago.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error in retrieve_trade_dates: {e}")
            return "", ""

    def evaluate_market_sentiment(self):
        """
        Evaluates the market sentiment based on recent news headlines.

        Returns:
            Tuple[float, str]: Probability of the dominant sentiment and the sentiment label.
        """
        try:
            current_date, past_date = self.retrieve_trade_dates()
            news_headlines = self.alpaca_api.get_news_headlines(
                symbol=self.trading_symbol,
                start_date=past_date,
                end_date=current_date
            )
            if not news_headlines:
                return 0, "neutral"

            probabilities, sentiments = zip(*[self.sentiment_analyzer.analyze(headline) for headline in news_headlines])
            avg_probability = sum(probabilities) / len(probabilities)
            dominant_sentiment = max(set(sentiments), key=sentiments.count)
            return avg_probability, dominant_sentiment
        except Exception as e:
            print(f"Error in evaluate_market_sentiment: {e}")
            return 0, "neutral"

    def execute_trading_logic(self):
        """
        Executes the trading logic based on market sentiment.
        """
        try:
            available_cash, current_price, trade_quantity = self.calculate_position_size()
            probability, market_sentiment = self.evaluate_market_sentiment()

            if available_cash > current_price:
                if market_sentiment == "positive" and probability > 0.999:
                    if self.previous_trade == "sell":
                        self.sell_all()
                    order = self.create_order(
                        self.trading_symbol,
                        trade_quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=current_price * 1.20,
                        stop_loss_price=current_price * 0.95
                    )
                    self.submit_order(order)
                    self.previous_trade = "buy"
                elif market_sentiment == "negative" and probability > 0.99:
                    if self.previous_trade == "buy":
                        self.sell_all()
                    order = self.create_order(
                        self.trading_symbol,
                        trade_quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=current_price * 0.8,
                        stop_loss_price=current_price * 1.05
                    )
                    self.submit_order(order)
                    self.previous_trade = "sell"
        except Exception as e:
            print(f"Error in execute_trading_logic: {e}")

# Main logic
if __name__ == "__main__":
    # Backtesting setup
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    broker = Alpaca({
        "API_KEY": API_KEY,
        "API_SECRET": API_SECRET,
        "PAPER": True
    })
    trading_strategy = AutomatedTradingStrategy(
        name='AutomatedMLTrading',
        broker=broker,
        parameters={"trading_symbol": "SPY", "investment_fraction": 0.5}
    )
    trading_strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={
            "trading_symbol": "SPY",
            "investment_fraction": 0.5
        }
    )
