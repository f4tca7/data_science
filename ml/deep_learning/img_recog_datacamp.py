import pandas as pd
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import datasets
from keras import backend as K

img_rows, img_cols = 28, 28
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 784)
# X_test = X_test.reshape(X_test.shape[0], 784)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# print(X_train.shape)
# quit()

model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(Conv2D(10, kernel_size=3, activation='relu'))           
model.add(MaxPooling2D(pool_size=(2, 2)))     
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=.2, epochs=3)
print(model.evaluate(X_test, y_test))