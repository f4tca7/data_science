import pandas as pd
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
model.add(Conv2D(5, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(2, kernel_size=3, activation='relu'))           
# model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)
callbacks_list = [checkpoint]
model.summary()
training = model.fit(X_train, y_train, validation_split=.2, epochs=3, batch_size=10, callbacks=callbacks_list)
print(model.evaluate(X_test, y_test, batch_size=10))
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
model.predict(X_test[0:3])
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()