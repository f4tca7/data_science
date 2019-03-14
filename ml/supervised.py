import pandas as pd

def fill_with_mean(data, party, col_name):
    data = data.loc[data['party'] == party]
    count_y = len(data[data.loc[:,col_name] == 'y'])
    count_n = len(data[data.loc[:,col_name] == 'n'])

    if count_y > count_n :
        data.loc[:,col_name] = data.loc[:,col_name].apply(lambda x: 'y' if x == '?' else x)
        #data[data.iloc[:,col_id] == '?'] = 'y'
    else :
        data.loc[:,col_name] = data.loc[:,col_name].apply(lambda x: 'y' if x == '?' else x)

    return data

votes = pd.read_csv('../datasets/house-votes-84.csv')
votes.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']

for i in range(len(votes.columns)) :
    if i > 0 :
        col_name = votes.columns[i]
        filled_data = fill_with_mean(votes[['party', col_name]], 'republican', col_name)
        votes.loc[votes['party'] == 'republican', col_name] = filled_data.loc[:, col_name]
        
        filled_data = fill_with_mean(votes[['party', col_name]], 'democrat', col_name)
        votes.loc[votes['party'] == 'democrat', col_name] = filled_data.loc[:, col_name]

print(votes)