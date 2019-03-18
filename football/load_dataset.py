import requests
import secrets
import datetime
import io
import numpy as np
import pandas as pd

MIN_SEASON = 1993
MAX_SEASON = 2018

def write_football_data(df, from_season, to_season, division) :
    filename = 'bundesliga_' + division + '_' + str(from_season) + '_' + str(to_season) + '.csv'
    filepath = '../datasets/football/' + filename
    df.to_csv(filepath)
    print('Created ' + filepath)

def load_football_data(from_season, to_season, division = 'D1'):
    """
    sample_url = 'http://www.football-data.co.uk/mmz4281/1819/D1.csv'
    from_season and to_season both specify the year in which the season began
    to load e.g. season 93/94 and 94/95, set from_season to 93 and to_season to 94
    """

    if from_season > to_season :
        raise Exception('from_season needs to be smaller or equal to to_season') 
    if (from_season > MAX_SEASON) | (from_season < MIN_SEASON) | (to_season > MAX_SEASON) | (to_season < MIN_SEASON) :
        raise Exception('from_season and to_season need to be in the range from ' + str(MIN_SEASON) + ' to ' + str(MAX_SEASON))  
    
    url_base = 'http://www.football-data.co.uk/mmz4281/'
    num_season = to_season - from_season + 1
    curr_season = from_season
    url = ''
    division_df = pd.DataFrame()
    for season in range(num_season):        
        if curr_season > MAX_SEASON :
            raise Exception('exceeded MAX_SEASON')

        from_season_str = str(curr_season)[-2:]      
        curr_to_season = str(curr_season + 1)[-2:]
        season_str = str(from_season_str) + str(curr_to_season)
        url = url_base + season_str + '/'
        url += division 
        url += '.csv'
        print('Fetching ' + url)
        resp = requests.get(url)
        if resp.status_code != 200:
            # This means something went wrong.
            raise Exception('GET ' + url + ' {}'.format(resp.status_code))
        else :
            print('200 GET ' + url)
            data = resp.content                
            c = pd.read_csv(io.StringIO(data.decode('ISO-8859-1')), parse_dates=True, error_bad_lines=False)
            c = c.dropna(how='all')
            c = c.iloc[:,:34]
            if division_df.empty :
                division_df = c
            else :
                division_df = pd.concat([division_df, c], sort=False)

        curr_season += 1
    return division_df

from_season = 1993
to_season = 2018

division = 'D2'
df = load_football_data(from_season, to_season, division)        
write_football_data(df, from_season, to_season, division)