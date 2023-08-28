import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

# Load the Data
df = pd.read_csv("cleaned_stock_data.csv")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row(dbc.Col(html.H1("Stock Analysis Dashboard"), width={'size': 6, 'offset': 3}), className='mb-4'),

    dbc.Row([
        dbc.Col([
            html.Label("Select Stock:"),
            dcc.Dropdown(id="stock-dropdown",
                         options=[{'label': stock, 'value': stock} for stock in df['name'].unique()],
                         value=df['name'].unique()[0],
                         clearable=False)
        ], width=4),

        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(id="date-picker",
                                start_date=df['date'].min(),
                                end_date=df['date'].max(),
                                display_format='YYYY-MM-DD')
        ], width=4)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-plot'), width=12),
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='moving-average-plot'), width=12), # New Moving Average Plot
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='volume-chart'), width=6),
        dbc.Col(html.Div(id='stats-display'), width=6)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='histogram'), width=6),
        dbc.Col(dcc.Graph(id='correlation-matrix'), width=6)
    ], className='mb-4')
])


@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('moving-average-plot', 'figure'), # New Output
     Output('volume-chart', 'figure'),
     Output('stats-display', 'children'),
     Output('histogram', 'figure'),
     Output('correlation-matrix', 'figure')],
    [Input('stock-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graphs(stock_selected, start_date, end_date):
    dff = df[(df['name'] == stock_selected) & (df['date'] >= start_date) & (df['date'] <= end_date)]

    # Time Series Plot
    ts_fig = go.Figure()
    for column in ['open', 'close', 'high', 'low']:
        ts_fig.add_trace(go.Scatter(x=dff['date'], y=dff[column],
                                    mode='lines', name=column.capitalize()))
    ts_fig.update_layout(title_text=f"Time Series of Stock {stock_selected}",
                         xaxis_title="Date", yaxis_title="Price")

    # Moving Average Plot
    ma_short = dff['close'].rolling(window=50).mean()
    ma_long = dff['close'].rolling(window=200).mean()
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=dff['date'], y=dff['close'],
                                mode='lines', name='Close Price'))
    ma_fig.add_trace(go.Scatter(x=dff['date'], y=ma_short,
                                mode='lines', name='50-day MA'))
    ma_fig.add_trace(go.Scatter(x=dff['date'], y=ma_long,
                                mode='lines', name='200-day MA'))
    ma_fig.update_layout(title_text=f"Moving Averages for Stock {stock_selected}",
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

    return ts_fig, ma_fig, volume_fig, stats_display, histogram, correlation_fig  # Added ma_fig here

if __name__ == '__main__':
    app.run_server(debug=True)
