import yfinance as yf

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()

    def fetch_data(self):
        data = yf.Ticker(self.ticker)
        return data.history(period="1mo")

    def is_current_price_lowest(self):
        if self.data.empty:
            return False

        current_price = self.data['Close'].iloc[-1]
        monthly_low = self.data['Low'].min()
        return current_price <= monthly_low

    def get_average_volume(self):
        return self.data['Volume'].mean()

    def had_high_volume_day(self):
        avg_volume = self.get_average_volume()
        return (self.data['Volume'] > 1.5 * avg_volume).any()

    def get_suggestion(self):
        if self.data.empty:
            return f"No data available for {self.ticker}."

        suggestion_str = ""

        if self.is_current_price_lowest():
            suggestion_str += f"Consider buying {self.ticker}. It's currently at its monthly low!"
        else:
            suggestion_str += f"{self.ticker} is not at its monthly low."

        if self.had_high_volume_day():
            suggestion_str += f" Note: {self.ticker} had a day with unusually high trading volume in the past month. This could indicate significant news or events."

        return suggestion_str
