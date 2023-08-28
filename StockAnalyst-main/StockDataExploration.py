import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class StockDataExploration:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def univariate_advanced_analysis(self):
        columns = ['open', 'high', 'low', 'close', 'volume']

        # Distribution Plots with Normal Distribution fit
        for column in columns:
            plt.figure(figsize=(10, 6))
            sns.distplot(self.df[column], fit=norm, kde=False)
            plt.title(f'Distribution plot of {column} with Normal Distribution')
            plt.show()

        # Violin plots
        for column in columns:
            plt.figure(figsize=(10, 6))
            sns.violinplot(self.df[column])
            plt.title(f'Violin plot of {column}')
            plt.show()

    def multivariate_advanced_analysis(self):
        columns = ['open', 'high', 'low', 'close', 'volume']

        # Joint plots
        sns.jointplot(x="open", y="volume", data=self.df, kind='scatter', height=8)
        plt.title('Joint plot of Opening Prices vs Volume')
        plt.show()

        # Pivot table for understanding stock patterns over time
        pivot_open = self.df.pivot_table(index='name', columns='data', values='open')
        plt.figure(figsize=(20, 10))
        sns.heatmap(pivot_open, cmap='coolwarm', linecolor='white', linewidth=1)
        plt.title('Heatmap of Opening Prices over Time')
        plt.show()

        # Cluster map
        sns.clustermap(pivot_open, cmap='coolwarm', standard_scale=1)
        plt.title('Clustered Heatmap of Opening Prices over Time')
        plt.show()

    def process_advanced_exploration(self):
        self.univariate_advanced_analysis()
        self.multivariate_advanced_analysis()

explorer = StockDataExploration(
    "C:\\Users\\User\\PycharmProjects\\StockPredictor\\StockAnalyst-main\\cleaned_stock_data.csv")
explorer.process_advanced_exploration()
