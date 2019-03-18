import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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


df_div1 = pd.read_csv('../datasets/football/bundesliga_D1_1993_2018.csv', parse_dates=True)
df_div1 = df_div1.dropna(how='all', axis=1)
df_div1 = df_div1.drop('Unnamed: 0', axis=1)
df_subset = df_div1[['season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
df_subset.columns = ['season', 'team_1', 'team_2', 'score_1', 'score_2', 'ftr']
df_subset['won'] = df_subset.loc[:,'ftr'].apply(lambda x : 0 if (x == 'D') | (x == 'A') else 1)
df_subset['score_diff'] = df_subset['score_2'] - df_subset['score_1']
df_subset['home'] = 1 
df_subset['team_1'] = df_subset['team_1'].replace('.','')
df_subset['team_2'] = df_subset['team_2'].replace('.','')
df_subset = df_subset.drop('ftr', axis=1)
teams = np.unique(df_subset['team_2'])
team_nums = np.arange(0, teams.shape[0], 1)
teams_dict = dict(zip(teams, team_nums))

df_subset['team_1'] = df_subset['team_1'].apply(lambda x : teams_dict[x])
df_subset['team_2'] = df_subset['team_2'].apply(lambda x : teams_dict[x])

y = df_subset['won'].values
X = df_subset[['team_1', 'team_2']].values
# X = df_subset['score_diff'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 3),
              "min_samples_leaf": randint(1, 3),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# # Create the hyperparameter grid
# c_space = np.logspace(-5, 8, 15)
# param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# # Instantiate the logistic regression classifier: logreg
# logreg = LogisticRegression()

# # Instantiate the GridSearchCV object: logreg_cv
# logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# # Fit it to the training data
# logreg_cv.fit(X_train, y_train)

# # Print the optimal parameters and best score
# print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
# print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('SVM', clf)]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))     