import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
import gzip
import gensim 
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MSFT_KEYOWRDS =  "microsoft|msft|nadella|gates|windows|cloud|azure|productivity"
GS_KEYWORDS = "goldman|sachs|bank|investment|speculation|security|banking|currrency|commoditiy|lend|invest|bubble"

def combine_nyt_data(filename_1, filename_2, relative_path, from_y, from_m, to_y, to_m):    
    """
    Utility function to combine two dataframes from the NYT API 
    """
    nyt_data_1 = pd.read_csv(relative_path + filename_1, index_col='date', parse_dates=True)
    nyt_data_2 = pd.read_csv(relative_path + filename_2, index_col='date', parse_dates=True)
    nyt_data = pd.concat([nyt_data_1, nyt_data_2], sort=False).drop_duplicates()
    filename = 'nyt_archive_' + str(from_y) + '_' + str(from_m)  + '_' + str(to_y) + '_' + str(to_m) + '.csv'
    full_path = relative_path + filename
    nyt_data.to_csv(full_path)
    print('Created ' + full_path)

def combine_text_columns(data_frame, to_drop):
    """ 
    Helper function to convert all text rows of data_frame to single vector 
    """
    
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)    
    text_data.fillna(' ', inplace=True)

    return text_data.apply(lambda x: " ".join(x), axis=1)


def preprocess(path_stocks, path_nyt, save_preprocessed=False):
    """
    Main preprocessing function
    """
    stemmer = WordNetLemmatizer()

    stocks = pd.read_csv(path_stocks, parse_dates=True, index_col='Date')
    stocks = stocks.shift(periods=-1)
    stocks = stocks.dropna()
    stocks.reset_index(inplace = True)
    stocks.loc[:,'Date'] = pd.to_datetime(stocks.loc[:,'Date'])

    nyt_data = pd.read_csv(path_nyt, parse_dates=True)

    nyt_data.loc[:,'date'] = pd.to_datetime(nyt_data.loc[:,'date'])

    stocks['went_up'] = (stocks['Close'] - stocks['Open']) / stocks['Open']
    stocks['went_up'] = stocks['went_up'].apply(lambda x: 1 if x > 0.005 else 0)

    stocks = stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close'], axis=1)
    stocks.columns = ['date', 'went_up']
    
    full_df = nyt_data.merge(stocks, on='date',suffixes=('','_y'))
    #full_df = nyt_data.merge(stocks, left_index=True, right_index=True)
    num_days = np.unique(full_df.index.values)
    print(full_df.head())
    print(full_df.columns)
 
    full_df = full_df.dropna()
    #full_df = full_df.drop('Unnamed: 0', axis=1)
    print('Full DF shape: ' + str(full_df.shape))
    print('Days: ' + str(len(num_days)))
    print('Positive Days: ' + str(len(np.unique(full_df.loc[full_df['went_up'] == 1].index.values))))

    full_df = full_df.reset_index(drop=True)
    print('Combining text into one column')
    full_df['all_text'] = combine_text_columns(full_df, ['went_up', 'date'])
    full_df = full_df.drop(['headline', 'snippet', 'keywords'], axis=1)



    print('Normalizing Text')
    # Remove all the special characters
    full_df['all_text'] = full_df['all_text'].apply(lambda x : re.sub(r'\W', ' ', x))
    # # remove all single characters
    full_df['all_text'] = full_df['all_text'].apply(lambda x : re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

    # Remove single characters from the start
    full_df['all_text'] = full_df['all_text'].apply(lambda x : re.sub(r'\^[a-zA-Z]\s+', ' ', x)) 

    # Substituting multiple spaces with single space
    full_df['all_text'] = full_df['all_text'].apply(lambda x : re.sub(r'\s+', ' ', x, flags=re.I))

    # Removing prefixed 'b'
    full_df['all_text'] = full_df['all_text'].apply(lambda x : re.sub(r'^b\s+', '', x))

    # Converting to Lowercase
    full_df['all_text'] = full_df['all_text'].apply(lambda x : x.lower())

    # Lemmatization
    full_df['all_text'] = full_df['all_text'].apply(lambda x : x.split())

    full_df['all_text'] = full_df['all_text'].apply(lambda x : [stemmer.lemmatize(word) for word in x])
    full_df['all_text'] = full_df['all_text'].apply(lambda x : ' '.join(x))

    print('Combine rows for same date')
    full_df['all_text'] = full_df.groupby(['date', 'went_up'],as_index=False)['all_text'].transform(lambda x: ' '.join(x))
    #full_df = full_df.drop(['index'], axis=1)
    full_df = full_df[['date','went_up','all_text']].drop_duplicates()
    #full_df.drop_duplicates(inplace = True)
    full_df = full_df.reset_index()

    filter_str = GS_KEYWORDS
    #print('Filtering rows for ' + filter_str)    
    #full_df = full_df[full_df['all_text_processed'].str.contains(filter_str)]
    print(full_df.head())
    print(full_df.shape)

    if save_preprocessed == True :
        rel_path = '../datasets/stock_data/'
        filename = 'preprocessed_nyt_stock_MSFT.csv'
        full_path = rel_path + filename
        full_df.to_csv(full_path)        
        print('Created ' + full_path)

    return full_df

def prep_vw_data(df):
    for line in df['all_text'].values: 
        
        #if (i%100==0):
        #    logging.info ("read {0} lines".format (i))
        # do some pre-processing and return a list of words for each review text
        yield gensim.utils.simple_preprocess(line)    

def train_vw(df):

    documents = list(prep_vw_data(df))
    logging.info ("Done reading data file")
    model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.save('v2w.model')

def load_vw():
    model = gensim.models.Word2Vec.load('v2w.model')
    w1 = "goldman"
    print(model.wv.most_similar(positive=w1,topn=100))

name1 = 'nyt_archive_2010_1_2016_1.csv'
name2 = 'nyt_archive_2016_1_2019_2.csv'
rel_path = '../datasets/large_data/'
#combine_nyt_data(name1, name2, rel_path, 2010, 1, 2019, 2)
stocks_path = '../datasets/stock_data/MSFT_2016_12_2019_2.csv'
nyt_path = '../datasets/large_data/nyt_archive_2016_1_2019_2.csv'
#df = preprocess(stocks_path, nyt_path, save_preprocessed=True)    
#df = pd.read_csv('../datasets/large_data/preprocessed_nyt_stock.csv', parse_dates=True)
#print(df.head())
#train_vw(df)
load_vw()