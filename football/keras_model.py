#       season  team_1  team_2  home  seed_diff  score_diff  score_1  score_2  \
# 3430    2012    1907     100     0          1           1       58       57   
# 3431    2012    1704     649     0         -8         -17       63       80   
# 3432    2012    4291     649     0          2          12       82       70   
# 3433    2012    7533     649     0        -11          -8       60       68   
# 3434    2012   10813     649     0         -7          -5       70       75   

#       won  
# 3430    1  
# 3431    0  
# 3432    1  
# 3433    0  
# 3434    0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

df_div1 = pd.read_csv('../datasets/football/bundesliga_D1_1993_2018.csv', parse_dates=True)
df_div1 = df_div1.dropna(how='all', axis=1)
df_div1 = df_div1.drop('Unnamed: 0', axis=1)
df_subset = df_div1[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
df_subset.columns = ['date', 'team_1', 'team_2', 'score_1', 'score_2', 'ftr']
df_subset['won'] = df_subset.loc[:,'ftr'].apply(lambda x : 0 if (x == 'D') | (x == 'A') else 1)
df_subset['score_diff'] = df_subset['score_2'] - df_subset['score_1']
df_subset = df_subset.drop('ftr', axis=1)
teams = np.unique(df_subset['team_1'])
team_nums = np.arange(0, teams.shape[0], 1)
teams_dict = dict(zip(teams, team_nums))

df_subset['team_1'] = df_subset['team_1'].apply(lambda x : teams_dict[x])
df_subset['team_2'] = df_subset['team_2'].apply(lambda x : teams_dict[x])
df_subset['date'] = pd.to_datetime(df_subset['date']).dt.year
print(df_subset.head())
