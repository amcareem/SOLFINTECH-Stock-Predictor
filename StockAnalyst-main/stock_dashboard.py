import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import tensorflow as tf

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class StockPredictor:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def split_data(self, stock_name, test_size=0.2):
        stock_data = self.data[self.data['name'] == stock_name]
        X = stock_data[['open', 'high', 'low', 'volume']]
        y = stock_data['close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def predict_future(self, stock_name, days):
        _, _, _, y_test = self.split_data(stock_name)
        y_test = y_test.reset_index(drop=True)
        model = ARIMA(y_test, order=(5, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=days)
        return predictions.values

    def get_current_prices(self):
        return {name: self.data[self.data['name'] == name]['close'].iloc[-1] for name in stock_names}

    def highest_profit_potential(self, amount):
        current_prices = self.get_current_prices()
        affordable_stocks = [stock for stock, price in current_prices.items() if price <= amount]
        profits = {}
        forecast_values = {}

        for stock_name in affordable_stocks:
            predicted_values = self.predict_future(stock_name, 30)
            profit = (predicted_values[-1] - predicted_values[0]) * (amount / predicted_values[0])
            profits[stock_name] = profit
            forecast_values[stock_name] = {
                '1_day': predicted_values[1] - predicted_values[0],
                '7_days': predicted_values[7] - predicted_values[0],
                '30_days': predicted_values[29] - predicted_values[0]
            }

        best_stock = max(profits, key=profits.get)
        return best_stock, profits[best_stock], forecast_values[best_stock]


def display_forecast(value, label):
    color = 'green' if value > 0 else 'red'
    action = 'Profit' if value > 0 else 'Loss'
    st.markdown(f"""
                <div style='border: 2px solid {color}; padding: 10px; border-radius: 10px;'>
                    <h3 style='color: {color}'>{label}</h3>
                    <p><b>Status:</b> {action} of <b>{abs(value):.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True)

def main():
    st.title("Stock Predictor Dashboard")
    st.sidebar.title("Settings")
    scenario = st.sidebar.radio("Choose Scenario",
                                ["Predict Future Stock Price", "Determine Best Stock for Investment"])

    if scenario == "Predict Future Stock Price":
        selected_stock = st.sidebar.selectbox("Select a Stock for Analysis", stock_names)
        st.header("Scenario 1: Predict Stock Price for Future Days")

        if st.button("Predict Future Values for " + selected_stock):
            predicted_values = predictor.predict_future(selected_stock, 30)
            one_day_profit = predicted_values[1] - predicted_values[0]
            seven_days_profit = predicted_values[7] - predicted_values[0]
            thirty_days_profit = predicted_values[29] - predicted_values[0]

            display_forecast(one_day_profit, '1-day forecast')
            display_forecast(seven_days_profit, '7-day forecast')
            display_forecast(thirty_days_profit, '30-day forecast')

    elif scenario == "Determine Best Stock for Investment":
        st.header("Scenario 2: Determine Best Stock for Investment")
        user_amount = st.number_input("Enter an amount", min_value=0.0, step=100.0)
        current_prices = predictor.get_current_prices()
        affordable_stocks = [stock for stock, price in current_prices.items() if price <= user_amount]

        if len(affordable_stocks) == 0:
            st.write(f"No stocks are available within the budget of {user_amount:.2f}")
        elif st.button("Find Best Stock for Investment"):
            best_stock, profit, forecast = predictor.highest_profit_potential(user_amount)
            st.write(f"Investing in {best_stock} has the highest potential profit of {profit:.2f} in the next 30 days.")

            st.write("### Forecast:")
            forecast_periods = ['1_day', '7_days', '30_days']
            labels = ['1 Day', '7 Days', '1 Month']

            for period, label in zip(forecast_periods, labels):
                value = forecast[period]
                display_forecast(value, label)


predictor = StockPredictor(r'C:\Users\User\PycharmProjects\StockPredictor\StockAnalyst-main\cleaned_stock_data.csv')
stock_names = ["AAPL", "ADBE", "AMZN", "BRK.A", "CSCO", "DIS", "ECL", "GEO", "GOOGL", "HSBC", "IBM", "INTC", "JNJ",
               "KO",
               "MCD", "META", "MSFT", "NVDA", "ORCL", "PEP", "QCOM", "RYAAY", "SHEL", "TM", "TSLA", "UPS", "VZ", "WFC",
               "WMT", "YELP"]

if __name__ == '__main__':
    main()
