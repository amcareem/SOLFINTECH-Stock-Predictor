import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class StockDataCleaner:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def handle_missing_data(self):
        # Drop rows with missing values
        self.df.dropna(inplace=True)

    def check_data_types(self):
        # Ensure the columns are of the correct data type
        self.df['date'] = pd.to_datetime(self.df['date'])
        for column in ['open', 'high', 'low', 'close', 'volume']:
            self.df[column] = pd.to_numeric(self.df[column])

    def visualize_outliers(self, columns=['open', 'high', 'low', 'close']):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[columns])
        plt.title('Boxplot of Stock Values')
        plt.show()

    def handle_outliers(self, columns=['open', 'high', 'low', 'close']):
        # Use Z-Score or IQR method here to handle outliers if needed.
        pass  # Placeholder for any outlier handling logic

    def save_cleaned_data(self, save_path="cleaned_stock_data.csv"):
        self.df.to_csv(save_path, index=False)

    def process_data(self):
        self.handle_missing_data()
        self.check_data_types()
        self.visualize_outliers()
        self.handle_outliers()
        self.save_cleaned_data()

cleaner = StockDataCleaner(
    "C:\\Users\\User\\PycharmProjects\\StockPredictor\\StockAnalyst-main\\consolidated_stock_data.csv")
cleaner.process_data()