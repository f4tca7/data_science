import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/exercises/titanic_all_numeric.csv')
predictors =  df.drop('survived', axis=1).as_matrix()

# Convert the target to categorical: target
target = to_categorical(df.survived)

#X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.4, random_state=42)

# Set up the model
n_cols = predictors.shape[1]
model_1 = Sequential()
input_shape = (n_cols,)
model_1.add(Dense(10, activation='relu', input_shape = input_shape))
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(2, activation='softmax'))

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
predictions = model_1.predict(predictors)
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
# compare = pd.DataFrame(columns=['test', 'pred', 'result'])
# compare['test'] = target[:,1]
# compare['pred'] = predicted_prob_true
# compare['result'] = False
# compare['result'] = (compare['test'] == 1.0) & (compare['pred'] > 0.5)
# compare['result'] = (compare['test'] == 0.0) & (compare['pred'] <= 0.5)
# print(compare.shape)
# print(compare.loc[compare.loc[:,'result'] == True].shape)

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))