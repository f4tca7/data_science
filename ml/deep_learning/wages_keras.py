import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential

df = pd.read_csv('../datasets/hourly_wages.csv')

predictors = df.drop('wage_per_hour', axis=1).values
target = df['wage_per_hour'].values

# Specify the model
n_cols = predictors.shape[1]
print(n_cols)
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target, epochs=10)