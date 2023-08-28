import unittest

import pandas as pd

from app import app, DataPreprocessor, StockAnalyzer

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app()
        self.client = self.app.test_client()

    def test_app_initialization(self):
        self.assertIsNotNone(self.app)

    def test_home_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_about_route(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)

    def test_stock_suggestions_route(self):
        response = self.client.get('/stock_suggestions')
        self.assertEqual(response.status_code, 200)

    def test_data_preprocessing_missing_data(self):
        data = {
            'numeric_col1': [1, 2, None, 4, 5],
            'numeric_col2': [None, 1, 3, 4, 5]
        }
        df = pd.DataFrame(data)
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.handle_missing_data(df)
        self.assertFalse(processed_data.isnull().any().any())

    def test_stock_analyzer_is_current_price_lowest(self):
        analyzer = StockAnalyzer('AAPL')
        self.assertIn(analyzer.is_current_price_lowest(), [True, False])

    def test_stock_analyzer_get_suggestion(self):
        analyzer = StockAnalyzer('AAPL')
        self.assertIsNotNone(analyzer.get_suggestion())

if __name__ == '__main__':
    unittest.main()
