import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def combine_nyt_data(filename_1, filename_2, relative_path, from_y, from_m, to_y, to_m):    
    nyt_data_1 = pd.read_csv(relative_path + filename_1, index_col='date', parse_dates=True)
    nyt_data_2 = pd.read_csv(relative_path + filename_2, index_col='date', parse_dates=True)
    nyt_data = pd.concat([nyt_data_1, nyt_data_2], sort=False).drop_duplicates()
    filename = 'nyt_archive_' + str(from_y) + '_' + str(from_m)  + '_' + str(to_y) + '_' + str(to_m) + '.csv'
    full_path = relative_path + filename
    nyt_data.to_csv(full_path)
    print('Created ' + full_path)

# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop):
    """ converts all text rows of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # Replace nans with blanks
    text_data.fillna(' ', inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

def preprocess(path_stocks, path_nyt, save_preprocessed):
    stemmer = WordNetLemmatizer()

    stocks = pd.read_csv(path_stocks, index_col='Date', parse_dates=True)
    stocks = stocks.shift(periods=-1)
    stocks = stocks.dropna()

    nyt_data = pd.read_csv(path_nyt, index_col='date', parse_dates=True)

    stocks['went_up'] = (stocks['Close'] - stocks['Open']) / stocks['Open']
    stocks['went_up'] = stocks['went_up'].apply(lambda x: 1 if x > 0.005 else 0)

    stocks = stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close'], axis=1)
    full_df = nyt_data.merge(stocks, left_index=True, right_index=True)
    num_days = np.unique(full_df.index.values)


    full_df = full_df.dropna()
    print('Full DF shape: ' + str(full_df.shape))
    print('Days: ' + str(len(num_days)))
    print('Positive Days: ' + str(len(np.unique(full_df.loc[full_df['went_up'] == 1].index.values))))

    full_df = full_df.reset_index(drop=True)
    print('Combining text into one column')
    full_df['all_text'] = combine_text_columns(full_df, ['went_up'])
    full_df = full_df.drop(['headline', 'snippet', 'keywords'], axis=1)

    full_df['headline_processed'] = ' '

    print('Normalizing Text')

    # Remove all the special characters
    full_df['all_text_processed'] = full_df['all_text'].apply(lambda x : re.sub(r'\W', ' ', x))
    # # remove all single characters
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

    # Remove single characters from the start
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : re.sub(r'\^[a-zA-Z]\s+', ' ', x)) 

    # Substituting multiple spaces with single space
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : re.sub(r'\s+', ' ', x, flags=re.I))

    # Removing prefixed 'b'
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : re.sub(r'^b\s+', '', x))

    # Converting to Lowercase
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : x.lower())

    # Lemmatization
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : x.split())

    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : [stemmer.lemmatize(word) for word in x])
    full_df['all_text_processed'] = full_df['all_text_processed'].apply(lambda x : ' '.join(x))
    filter_str = "microsoft|msft|nadella|gates|windows|cloud|azure|productivity"
    print('Filtering rows for ' + filter_str)    
    full_df = full_df[full_df['all_text_processed'].str.contains(filter_str)]
    print(full_df.head())
    print(full_df.shape)

    if save_preprocessed == True :
        rel_path = '../datasets/stock_data/'
        filename = 'preprocessed_nyt_stock.csv'
        full_path = rel_path + filename
        full_df.to_csv(full_path)        
        print('Created ' + full_path)

    return full_df

name1 = 'nyt_archive_2010_1_2016_1.csv'
name2 = 'nyt_archive_2016_1_2019_2.csv'
rel_path = '../datasets/large_data/'
#combine_nyt_data(name1, name2, rel_path, 2010, 1, 2019, 2)
stocks_path = '../datasets/stock_data/MSFT_2010_1_2019_2.csv'
nyt_path = '../datasets/large_data/nyt_archive_2016_1_2019_2.csv'
#df = preprocess(stocks_path, nyt_path)    