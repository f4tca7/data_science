import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
import stocks_preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import keras
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import datasets
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import preprocess_nyt_data

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def nn(full_df):


    y = full_df['went_up'].values
    text = full_df['all_text_processed'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        text, y, test_size=0.2, random_state=22)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    lstm_output_size = 70
    maxlen = 100
    embedding_dim = 50
    pool_size = 4
    embedding_matrix = create_embedding_matrix(
        '../datasets/large_data/glove_word_embeddings/glove.6B.50d.txt',
        tokenizer.word_index, embedding_dim)

    
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)    


    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, 
                            weights=[embedding_matrix], 
                            input_length=maxlen, 
                            trainable=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(128, 5, activation='relu'))                       
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.LSTM(lstm_output_size))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=5,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=30)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)


def build_fit_model(full_df):

    #X = full_df['headline_processed'].values
    y = full_df['went_up'].values
    text = full_df['all_text_processed'].values

    print('Vectorize')
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1, 2))  
    X = vectorizer.fit_transform(text).toarray()  

    print('TfidfTransform')
    tfidfconverter = TfidfTransformer()  
    X = tfidfconverter.fit_transform(X).toarray()  

    print('train_test_split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)  
    print('Train Shape: ' + str(X_train.shape))
    print('Test Shape: ' + str(X_test.shape))

    classifier = LogisticRegression()
    #classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  

    #classifier = SVC(probability=True)
    #classifier = KNeighborsClassifier(n_neighbors = 10)

    print('fit')
    history = classifier.fit(X_train, y_train)  

    print('predict')
    y_pred = classifier.predict(X_test)  

    #y_pred2 = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))  
    print(accuracy_score(y_test, y_pred))  

    y_pred_prob = classifier.predict_proba(X_test)[:,1]
    # print(y_pred_prob)
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

    # # Plot ROC curve
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.show()

    advwords = vectorizer.get_feature_names()
    advcoeffs = classifier.coef_.tolist()[0]
    advcoeffdf = pd.DataFrame({'Words' : advwords, 
                            'Coefficient' : advcoeffs})
    advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
    print(advcoeffdf.head(10))

    # loss, accuracy = classifier.evaluate(X_train, y_train, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    # loss, accuracy = classifier.evaluate(X_test, y_test, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))
    #plot_history(history)
    # # Compute and print AUC score
    # print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # # Compute cross-validated AUC scores: cv_auc
    # cv_auc = cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')

    # # Print list of AUC scores
    # print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))



stocks_path = '../datasets/stock_data/MSFT_2000_1_2019_3.csv'
nyt_path = '../datasets/large_data/nyt_archive_2000_1_2019_1.csv'
nyt_preprod_path = '../datasets/large_data/nyt_preprocessed_2000_1_2019_1.csv'
#nyt_data = pd.read_csv(nyt_path, parse_dates=True)
#df = preprocess_nyt_data.preprocess(nyt_data, save_preprocessed=True, base_path='../datasets/large_data/', filename='nyt_preprocessed_2000_1_2019_1.csv')
nyt_data = pd.read_csv(nyt_preprod_path, parse_dates=True)
#stocks_preprocessing.train_vw(nyt_data)

stocks_preprocessing.load_vw()
# df = stocks_precprocessing.preprocess(stocks_path, nyt_path, save_preprocessed=True)
# df = pd.read_csv('../datasets/stock_data/preprocessed_nyt_stock_MSFT.csv', parse_dates=True)
# build_fit_model(df)
# nn(df)