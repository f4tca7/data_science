import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

votes = pd.read_csv('../datasets/house-votes-84.csv')
votes.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']

votes = votes.replace('y', 1)
votes = votes.replace('n', 0)
votes = votes.replace('republican', 0)
votes = votes.replace('democrat', 1)
votes[votes == '?'] = np.nan
print(votes.isnull().sum())

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Create arrays for the features and the response variable
target = votes['party'].values
target = to_categorical(target)
predictors = votes.drop('party', axis=1).as_matrix()
predictors = imp.fit_transform(predictors)


print(predictors)

n_cols = predictors.shape[1]
model_1 = Sequential()
input_shape = (n_cols,)
model_1.add(Dense(100, activation='relu', input_shape = input_shape))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(2, activation='softmax'))

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=40, validation_split=0.2, callbacks=[early_stopping_monitor])

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


predictions = model_1.predict(predictors)
predicted_prob_true = predictions[:,1]