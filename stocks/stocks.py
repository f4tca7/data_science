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


def build_fit_model(full_df):

    #X = full_df['headline_processed'].values
    y = full_df['went_up'].values

    print('Vectorize')
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1, 2))  
    X = vectorizer.fit_transform(full_df['all_text_processed']).toarray()  

    print('TfidfTransform')
    tfidfconverter = TfidfTransformer()  
    X = tfidfconverter.fit_transform(X).toarray()  

    print('train_test_split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)  
    print('Train Shape: ' + str(X_train.shape))
    print('Test Shape: ' + str(X_test.shape))

    classifier = LogisticRegression()
    #classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  

    #classifier = SVC(probability=True)
    #classifier = KNeighborsClassifier(n_neighbors = 10)

    print('fit')
    classifier.fit(X_train, y_train)  

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

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    advwords = vectorizer.get_feature_names()
    advcoeffs = classifier.coef_.tolist()[0]
    advcoeffdf = pd.DataFrame({'Words' : advwords, 
                            'Coefficient' : advcoeffs})
    advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
    print(advcoeffdf.head(10))
    # # Compute and print AUC score
    # print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # # Compute cross-validated AUC scores: cv_auc
    # cv_auc = cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')

    # # Print list of AUC scores
    # print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

stocks_path = '../datasets/stock_data/MSFT_2010_1_2019_2.csv'
nyt_path = '../datasets/large_data/nyt_archive_2010_1_2019_2.csv'
# df = stocks_preprocessing.preprocess(stocks_path, nyt_path, save_preprocessed=True)
df = pd.read_csv('../datasets/stock_data/preprocessed_nyt_stock.csv', parse_dates=True)

build_fit_model(df)