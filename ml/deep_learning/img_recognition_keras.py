
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


from sklearn import datasets

# Create the model: model
model = Sequential()
digits = datasets.load_digits()
X = digits.data
y = to_categorical(digits.target)
print(y.shape)
print(X.shape)
print(type(X))

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape = (64,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model_1_training = model.fit(X, y, validation_split=0.3, epochs=20)
# model_1_training = model_1.fit(predictors, target, epochs=40, validation_split=0.2, callbacks=[early_stopping_monitor])

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
