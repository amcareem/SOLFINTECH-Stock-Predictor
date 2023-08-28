import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import imageio
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.scores = {}
        self.preprocess_data()

    def preprocess_data(self):
        self.data = self.data.sort_values(by=['name', 'date'])
        self.data.reset_index(drop=True, inplace=True)

    def train_test_split(self, stock_name):
        stock_data = self.data[self.data['name'] == stock_name]
        X = stock_data[['open', 'high', 'low', 'volume']].values
        y = stock_data['close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def plot_predictions(self, y_test, predictions, stock_name, model_name, save_path):
        plt.figure(figsize=(15, 7))
        plt.plot(y_test, color='blue', label='Actual Stock Price')
        plt.plot(predictions, color='red', linestyle='dashed', label='Predicted Stock Price')
        plt.title(f'{stock_name} Stock Price Prediction using {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def train_LinearRegression(self, stock_name, save_path):
        X_train, X_test, y_train, y_test = self.train_test_split(stock_name)
        model = LinearRegression().fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        if stock_name not in self.scores:
            self.scores[stock_name] = {}
        self.scores[stock_name]['Linear Regression'] = mse
        self.plot_predictions(y_test, predictions, stock_name, "Linear Regression", save_path)
        return predictions

    def train_ARIMA(self, stock_name, save_path):
        _, _, y_train, y_test = self.train_test_split(stock_name)
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(y_test))
        mse = mean_squared_error(y_test, predictions)
        self.scores[stock_name]['ARIMA'] = mse
        self.plot_predictions(y_test, predictions, stock_name, "ARIMA", save_path)
        return predictions

    def train_LSTM(self, stock_name, save_path):
        X_train, X_test, y_train, y_test = self.train_test_split(stock_name)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32)
        predictions = model.predict(X_test_scaled).squeeze()
        mse = mean_squared_error(y_test, predictions)
        self.scores[stock_name]['LSTM'] = mse
        self.plot_predictions(y_test, predictions, stock_name, "LSTM", save_path)
        return predictions

    def get_best_model(self, stock_name):
        return min(self.scores[stock_name], key=self.scores[stock_name].get)

    def ensemble_predictions(self, stock_name, lr_pred, arima_pred, lstm_pred):
        lr_weight = 1 / self.scores[stock_name]['Linear Regression']
        arima_weight = 1 / self.scores[stock_name]['ARIMA']
        lstm_weight = 1 / self.scores[stock_name]['LSTM']
        total_weight = lr_weight + arima_weight + lstm_weight
        return (lr_weight * lr_pred + arima_weight * arima_pred + lstm_weight * lstm_pred) / total_weight

if __name__ == "__main__":
    predictor = StockPredictor('cleaned_stock_data.csv')
    stock_names = ["AAPL", "ADBE", "AMZN", "BRK.A", "CSCO", "DIS", "ECL", "GEO", "GOOGL", "HSBC", "IBM", "INTC", "JNJ", "KO", "MCD", "META", "MSFT", "NVDA", "ORCL", "PEP", "QCOM", "RYAAY", "SHEL", "TM", "TSLA", "UPS", "VZ", "WFC", "WMT", "YELP"]
    saved_plots = []

    for stock_name in stock_names:
        print(f"Processing {stock_name}...")
        lr_plot_path = f"{stock_name}_LinearRegression_plot.png"
        arima_plot_path = f"{stock_name}_ARIMA_plot.png"
        lstm_plot_path = f"{stock_name}_LSTM_plot.png"
        ensemble_plot_path = f"{stock_name}_Ensemble_plot.png"
        saved_plots.extend([lr_plot_path, arima_plot_path, lstm_plot_path, ensemble_plot_path])
        lr_predictions = predictor.train_LinearRegression(stock_name, lr_plot_path)
        arima_predictions = predictor.train_ARIMA(stock_name, arima_plot_path)
        lstm_predictions = predictor.train_LSTM(stock_name, lstm_plot_path)
        ensemble_predictions = predictor.ensemble_predictions(stock_name, lr_predictions, arima_predictions, lstm_predictions)
        _, _, _, y_test = predictor.train_test_split(stock_name)
        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        predictor.scores[stock_name]['Ensemble'] = ensemble_mse
        predictor.plot_predictions(y_test, ensemble_predictions, stock_name, "Ensemble", ensemble_plot_path)
        best_model = predictor.get_best_model(stock_name)
        print(f"Best model for {stock_name} is {best_model} with MSE: {predictor.scores[stock_name][best_model]:.2f}")

    images = [imageio.imread(plot) for plot in saved_plots]
    fig, axs = plt.subplots(nrows=len(stock_names), ncols=4, figsize=(20, 5 * len(stock_names)))
    for i, stock_name in enumerate(stock_names):
        axs[i, 0].imshow(images[i * 4])
        axs[i, 1].imshow(images[i * 4 + 1])
        axs[i, 2].imshow(images[i * 4 + 2])
        axs[i, 3].imshow(images[i * 4 + 3])
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        axs[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig("combined_stock_predictions.png", dpi=300)
    plt.show()
    for plot in saved_plots:
        os.remove(plot)

        plt.show()

        for plot in saved_plots:
            os.remove(plot)

    model_performance_df = pd.DataFrame(predictor.scores).T
    model_performance_df.to_csv('model_performance_metrics.csv', index=True)


