import pandas as pd
from flask import Flask, render_template, request, redirect
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from flask import render_template_string
import yfinance as yf

app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

@app.route('/streamlit/')
def streamlit_app():
    return render_template('streamlit.html')

# Data Preprocessing Class
class DataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def handle_missing_data(self, df):
        try:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_columns] = self.numerical_imputer.fit_transform(df[numeric_columns])
        except Exception as e:
            print(f"Error handling missing data: {e}")
        return df

    def handle_outliers(self, df):
        try:
            numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Only select numeric columns
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ~((numeric_df >= (Q1 - 1.5 * IQR)) & (numeric_df <= (Q3 + 1.5 * IQR)))
            for col in numeric_df.columns:
                df[col].where(~outlier_condition[col], df[col].median(), inplace=True)
        except Exception as e:
            print(f"Error handling outliers: {e}")
        return df

    def scale_data(self, df):
        try:
            columns_to_scale = ['volume']  # Only scaling volume for now
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        except Exception as e:
            print(f"Error scaling data: {e}")
        return df

# Load and preprocess data with error handling
try:
    cleaned_stock_data = pd.read_csv("cleaned_stock_data.csv")
    cleaned_stock_data['date'] = pd.to_datetime(cleaned_stock_data['date'])
    preprocessor = DataPreprocessor()
    cleaned_stock_data = preprocessor.handle_missing_data(cleaned_stock_data)
    cleaned_stock_data = preprocessor.handle_outliers(cleaned_stock_data)
    cleaned_stock_data = preprocessor.scale_data(cleaned_stock_data)
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")

# Function to get current stock price with error handling
def get_current_price(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="1d")
        return data['Close'][0]
    except Exception as e:
        print(f"Error fetching price for {stock_symbol}: {e}")
        return None  # or some default value

# Dash layout
dash_app.layout = html.Div([
    dbc.Row(dbc.Col(html.H1("Stock Analysis Dashboard"), width={'size': 6, 'offset': 3}), className='mb-4'),

    dbc.Row(dbc.Col(html.Div(id='live-stock-price'), width={'size': 6, 'offset': 3}), className='mb-4'),

    dbc.Row([
        dbc.Col([
            html.Label("Select Stock:"),
            dcc.Dropdown(id="stock-dropdown",
                         options=[{'label': stock, 'value': stock} for stock in cleaned_stock_data['name'].unique()],
                         value=cleaned_stock_data['name'].unique()[0],
                         clearable=False)
        ], width=4),

        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(id="date-picker",
                                start_date=cleaned_stock_data['date'].min(),
                                end_date=cleaned_stock_data['date'].max(),
                                display_format='YYYY-MM-DD')
        ], width=4)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-plot'), width=12),
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='volume-chart'), width=6),
        dbc.Col(html.Div(id='stats-display'), width=6)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='histogram'), width=6),
        dbc.Col(dcc.Graph(id='correlation-matrix'), width=6)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='moving-average-plot'), width=12),
    ], className='mb-4')
])


@dash_app.callback(
    [Output('time-series-plot', 'figure'),
     Output('volume-chart', 'figure'),
     Output('stats-display', 'children'),
     Output('histogram', 'figure'),
     Output('correlation-matrix', 'figure'),
     Output('moving-average-plot', 'figure'),
     Output('live-stock-price', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graphs(stock_selected, start_date, end_date):
    dff = cleaned_stock_data[
        (cleaned_stock_data['name'] == stock_selected) & (cleaned_stock_data['date'] >= start_date) & (
                    cleaned_stock_data['date'] <= end_date)]

    # Fetching live price
    current_price = get_current_price(stock_selected)
    live_stock_price_display = f"Current Price of {stock_selected}: ${current_price:.2f}"

    # Time Series Plot
    ts_fig = go.Figure()
    for column in ['open', 'close', 'high', 'low']:
        ts_fig.add_trace(go.Scatter(x=dff['date'], y=dff[column],
                                    mode='lines', name=column.capitalize()))

    ts_fig.update_layout(title_text=f"Time Series of Stock {stock_selected}",
                         xaxis_title="Date", yaxis_title="Price")

    # Volume Chart
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(x=dff['date'], y=dff['volume'], name='Volume'))

    volume_fig.update_layout(title_text=f"Volume of Stock {stock_selected}",
                             xaxis_title="Date", yaxis_title="Volume")

    # Descriptive Statistics
    stats = dff[['open', 'high', 'low', 'close']].describe().transpose().round(2)
    stats_display = dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True)

    # Histogram
    histogram = go.Figure()
    histogram.add_trace(go.Histogram(x=dff['close'], name='Close Price'))
    histogram.update_layout(title_text=f"Distribution of Close Price for {stock_selected}",
                            xaxis_title="Price", yaxis_title="Frequency")

    # Correlation Matrix
    correlation = dff[['open', 'high', 'low', 'close']].corr()
    correlation_fig = ff.create_annotated_heatmap(z=correlation.values,
                                                  x=list(correlation.columns),
                                                  y=list(correlation.index),
                                                  annotation_text=correlation.round(2).values,
                                                  showscale=True)
    correlation_fig.update_layout(title_text=f"Correlation Matrix for {stock_selected}")

    # Moving Average Plot
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=dff['date'], y=dff['close'].rolling(window=5).mean(),
                                mode='lines', name='5-Day MA'))
    ma_fig.add_trace(go.Scatter(x=dff['date'], y=dff['close'].rolling(window=20).mean(),
                                mode='lines', name='20-Day MA'))
    ma_fig.update_layout(title_text=f"Moving Averages of Stock {stock_selected}",
                         xaxis_title="Date", yaxis_title="Price")

    return ts_fig, volume_fig, stats_display, histogram, correlation_fig, ma_fig, live_stock_price_display


# StockAnalyzer Class
class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()

    def fetch_data(self):
        data = yf.Ticker(self.ticker)
        return data.history(period="1mo")

    def is_current_price_lowest(self):
        if self.data.empty:  # Check if the dataframe is empty
            return False  # We can't determine if the price is the lowest if there's no data

        current_price = self.data['Close'].iloc[-1]  # Using iloc for safer indexing
        monthly_low = self.data['Low'].min()
        return current_price <= monthly_low

    def get_suggestion(self):
        if self.is_current_price_lowest():
            return f"Consider buying {self.ticker}. It's currently at its monthly low!"
        else:
            return f"{self.ticker} is not at its monthly low."


# New Flask route for Stock Suggestions
@app.route('/stock_suggestions')
def stock_suggestions():
    STOCK_LIST = [
        "AAPL", "ADBE", "AMZN", "BRK.A", "CSCO", "DIS", "ECL", "GEO", "GOOGL", "HSBC",
        "IBM", "INTC", "JNJ", "KO", "MCD", "META", "MSFT", "NVDA", "ORCL", "PEP",
        "QCOM", "RYAAY", "SHEL", "TM", "TSLA", "UPS", "VZ", "WFC", "WMT", "YELP"
    ]

    suggestions = []
    for ticker in STOCK_LIST:
        analyzer = StockAnalyzer(ticker)
        suggestions.append(analyzer.get_suggestion())

    return render_template_string('''
    <h1>Stock Buying Suggestions:</h1>
    <ul>
    {% for suggestion in suggestions %}
        <li>{{ suggestion }}</li>
    {% endfor %}
    </ul>
    ''', suggestions=suggestions)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/result', methods=['POST'])
def result():
    company_name = request.form['impath']
    interest_rate_change = float(request.form['interest_rate_change'])
    confidence_index = float(request.form['confidence_index'])

    prediction = get_current_price(company_name)


    # Apply adjustments to the prediction:
    adjusted_prediction = prediction
    adjusted_prediction -= prediction * (0.005 * interest_rate_change)
    adjusted_prediction += prediction * (0.01 * (confidence_index / 10))

    # Render the result with both the original and adjusted predictions.
    return render_template('result.html', original_prediction=prediction, adjusted_prediction=adjusted_prediction)


if __name__ == '__main__':
    app.run(debug=True)