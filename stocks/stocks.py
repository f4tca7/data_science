import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = pd.read_csv('../datasets/stock_data/MSFT_2018_12_2019_1.csv', index_col='Date', parse_dates=True)


nyt_data = pd.read_csv('../datasets/nyt_archive_2018_12_2019_1.csv', index_col='date', parse_dates=True)

stocks['went_up'] = (stocks['Adj Close'] - stocks['Open']) / stocks['Open']
stocks['went_up'] = stocks['went_up'].apply(lambda x: 1 if x > 0.005 else 0)
stocks = stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close'], axis=1)
print(stocks.head())

full_df = nyt_data.merge(stocks, left_index=True, right_index=True)
full_df = full_df.dropna()
