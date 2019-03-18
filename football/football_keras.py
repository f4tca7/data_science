import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input, Dense, Subtract, Embedding, Flatten, Concatenate
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils import plot_model


def plot_model_(model):
    # Plot the model
    plot_model(model, to_file='model.png')

    # Display the image
    data = plt.imread('model.png')
    plt.imshow(data)
    plt.show()

vectorizer = CountVectorizer()


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
X = df_subset[['team_1', 'team_2', 'home']].values
# X = df_subset['score_diff'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


n_teams = np.unique(df_subset['team_2']).shape[0]


team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')
# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')

# Input layer for team 1
team_in_1 = Input((1,), name="Team-1-In")

# Separate input layer for team 2
team_in_2 = Input((1,), name="Team-2-In")
home_in = Input(shape=(1,), name='Home-In')
# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)
# Subtraction layer from previous exercise
#score_diff = Subtract()([team_1_strength, team_2_strength])
conca = Concatenate()([team_1_strength, team_2_strength, home_in])

middle1 = Dense(1000, activation='relu')(conca)
middle2 = Dense(50, activation='relu')(middle1)
middle3 = Dense(50, activation='relu')(middle2)

out1 = Dense(1)(middle3)
# Create the model
model = Model([team_in_1, team_in_2, home_in], out1)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
input_1 = df_subset['team_1']
input_2 = df_subset['team_2']
model_training = model.fit([X_train[:,0], X_train[:,1], X_train[:,2]],
          y_train,
          epochs=100,
          batch_size=2048,
          validation_data=([X_test[:,0], X_test[:,1], X_test[:,2]], y_test),
          verbose=True)

# Get team_1 from the tournament data
input_1 = X_test[:,0]

# Get team_2 from the tournament data
input_2 = X_test[:,1]
input_3 = X_test[:,2]
# Evaluate the model using these inputs
model.evaluate([input_1, input_2, input_3], y_test)

# plot_model_(model)

plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

# input_tensor = Input(shape=(1,))
# output_tensor = Dense(1)(input_tensor)
# n_teams = unique(df_subset['team_1']).shape[0]
# team_lookup = Embedding(input_dim=n_teams,
#                         output_dim=1,
#                         input_length=1,
#                         name='Team-Strength')
# model = Model(input_tensor, output_tensor)
# model.compile(optimizer='adam', loss='mean_absolute_error')
# model.fit(X_train, y_train,
#           epochs=1,
#           batch_size=128,
#           validation_split=.1,
#           verbose=True)

# # Evaluate the model on the test data
# model.evaluate(X_test, y_test)