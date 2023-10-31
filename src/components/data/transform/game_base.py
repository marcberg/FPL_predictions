import pandas as pd 
import numpy as np
import os

def games_base():
    '''
    
    '''
    games = pd.read_csv(os.getcwd() + "/artifacts/fetched_data/get_game_list.csv")

    games['kickoff'] = pd.to_datetime(games['kickoff'])
    games['kickoff_date'] = games['kickoff'].dt.date
    games['kickoff_year'] = games['kickoff'].dt.year
    games['kickoff_month'] = games['kickoff'].dt.month
    games['rounds_left'] = 38-games['GW']
    games['label_1'] = np.where(games['team_h_score'] > games['team_a_score'], 1, 0)
    games['label_X'] = np.where(games['team_h_score'] == games['team_a_score'], 1, 0)
    games['label_2'] = np.where(games['team_h_score'] < games['team_a_score'], 1, 0)
    games['train_score'] = np.where(games['finished'], 'train', 'score')
    games['season'] = games['season_start_year'].astype(str)


    games_base = games[[
        # id
        'season_start_year', 
        'kickoff_date',
        'GW', 
        'id', 
        'team_h', 
        'team_a', 
        'train_score',

        # label
        'label_1', 
        'label_X', 
        'label_2', 

        # features
        'home',  
        'away', 
        'season',
        'kickoff_year', 
        'kickoff_month', 
        'rounds_left']]

    games_base.sort_values(['season_start_year', 'GW']).reset_index(drop=True)

    return games_base