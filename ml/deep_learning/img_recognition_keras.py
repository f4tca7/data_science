import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.datasets import mnist
from keras import backend as K

img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 12


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#print(X[1,:])
#print(mnist.iloc[105,0])
#img = np.reshape(X[105,:], (28, 28))

#print(img)
#imgplot = plt.imshow(img)
#plt.show()
#quit()
# Create the model: model

#digits = datasets.load_digits()
#X = digits.data
#y = to_categorical(digits.target)
# print(y.shape)
# print(X)
#print(X[1,:])
#print(y)
#quit()

model = Sequential()
# Add the first hidden layer
	
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))



model.add(Flatten(input_shape = input_shape))
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=5)
# Fit the model
model_1_training = model.fit(X_train, y_train, epochs=epochs, callbacks=[early_stopping_monitor], batch_size=batch_size, validation_data=(X_test, y_test))

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])