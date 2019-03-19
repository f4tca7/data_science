import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
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

stemmer = WordNetLemmatizer()

# nyt_1 = pd.read_csv('../datasets/large_data/intermediate_nyt_archive_2010_1_2017_1.csv', parse_dates=True)
# nyt_2 = pd.read_csv('../datasets/large_data/nyt_archive_2016_12_2019_2.csv', parse_dates=True)
# nyt_full = pd.concat([nyt_1, nyt_2], sort=False).drop_duplicates().reset_index(drop=True)
# nyt_full.to_csv('../datasets/large_data/' + 'nyt_archive_2010_12_2019_2.csv')



#stocks = pd.read_csv('../datasets/stock_data/MSFT_2018_12_2019_1.csv', index_col='Date', parse_dates=True)
stocks = pd.read_csv('../datasets/stock_data/MSFT_2010_1_2019_2.csv', index_col='Date', parse_dates=True)
stocks = stocks.shift(periods=-1)
stocks = stocks.dropna()

# nyt_data = pd.read_csv('../datasets/large_data/nyt_archive_2018_12_2019_1.csv', index_col='date', parse_dates=True)
# nyt_data = pd.read_csv('../datasets/large_data/nyt_archive_2016_12_2019_2.csv', index_col='date', parse_dates=True)
nyt_data = pd.read_csv('../datasets/large_data/nyt_archive_2010_1_2019_2.csv', index_col='date', parse_dates=True)

stocks['went_up'] = (stocks['Close'] - stocks['Open']) / stocks['Open']
stocks['went_up'] = stocks['went_up'].apply(lambda x: 1 if x > 0.005 else 0)
#print(np.unique(stocks.loc[stocks['went_up'] == 1].index.values))


stocks = stocks.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close'], axis=1)


full_df = nyt_data.merge(stocks, left_index=True, right_index=True)
num_days = np.unique(full_df.index.values)


full_df = full_df.dropna()
print('Full DF shape: ' + str(full_df.shape))
print('Days: ' + str(len(num_days)))
print('Positive Days: ' + str(len(np.unique(full_df.loc[full_df['went_up'] == 1].index.values))))

full_df = full_df.reset_index(drop=True)
full_df['headline_processed'] = ' '

# Remove all the special characters
full_df['headline_processed'] = full_df['headline'].apply(lambda x : re.sub(r'\W', ' ', x))
# # remove all single characters
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

# Remove single characters from the start
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : re.sub(r'\^[a-zA-Z]\s+', ' ', x)) 

# Substituting multiple spaces with single space
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : re.sub(r'\s+', ' ', x, flags=re.I))

# Removing prefixed 'b'
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : re.sub(r'^b\s+', '', x))

# Converting to Lowercase
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : x.lower())

# Lemmatization
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : x.split())

full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : [stemmer.lemmatize(word) for word in x])
full_df['headline_processed'] = full_df['headline_processed'].apply(lambda x : ' '.join(x))



#X = full_df['headline_processed'].values
y = full_df['went_up'].values

print('Vectorize')
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1, 2))  
X = vectorizer.fit_transform(full_df['headline_processed']).toarray()  

print('TfidfTransform')
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()  

print('train_test_split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)  
print('Train Shape: ' + str(X_train.shape))
print('Test Shape: ' + str(X_test.shape))

classifier = LogisticRegression()
#classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  

#classifier = SVC()
#classifier = KNeighborsClassifier(n_neighbors = 5)

print('fit')
classifier.fit(X_train, y_train)  

print('predict')
y_pred = classifier.predict(X_test)  


#y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
print(accuracy_score(y_test, y_pred))  

y_pred_prob = classifier.predict_proba(X_test)[:,1]
print(y_pred_prob)
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# # Compute and print AUC score
# print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# # Compute cross-validated AUC scores: cv_auc
# cv_auc = cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')

# # Print list of AUC scores
# print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
