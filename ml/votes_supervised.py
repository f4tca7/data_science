import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def fill_with_mean(data, party, col_name):
    data_loc = data.loc[data['party'] == party]
    count_y = len(data_loc[data_loc.loc[:,col_name] == 'y'])
    count_n = len(data_loc[data_loc.loc[:,col_name] == 'n'])

    if count_y > count_n :
        data_loc.loc[:,col_name] = data_loc.loc[:,col_name].apply(lambda x: 'y' if x == '?' else x)
        #data[data.iloc[:,col_id] == '?'] = 'y'
    else :
        data_loc.loc[:,col_name] = data_loc.loc[:,col_name].apply(lambda x: 'y' if x == '?' else x)

    return data_loc

votes = pd.read_csv('../datasets/house-votes-84.csv')
votes.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']

for i in range(len(votes.columns)) :
    if i > 0 :
        col_name = votes.columns[i]
        filled_data_1 = fill_with_mean(votes[['party', col_name]], 'republican', col_name)
        votes.loc[votes['party'] == 'republican', col_name] = filled_data_1.loc[:, col_name]
        
        filled_data_2 = fill_with_mean(votes[['party', col_name]], 'democrat', col_name)
        votes.loc[votes['party'] == 'democrat', col_name] = filled_data_2.loc[:, col_name]

votes = votes.replace('y', 1)
votes = votes.replace('n', 0)

# Create arrays for the features and the response variable
y = votes['party'].values
X = votes.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X, y)
y_pred = knn.predict(X)

X_new_vals = np.array([[ 0.5355251 ,  0.52268492,  0.30284059,  0.93824647,  0.93707859,
         0.71255703,  0.80564365,  0.36137492,  0.02447344,  0.99300338,
         0.86646522,  0.31951122,  0.30873045,  0.82214537,  0.24133158,
         0.05820214]])
X_new = pd.DataFrame(data = X_new_vals)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
