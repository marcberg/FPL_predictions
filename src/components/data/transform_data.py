import pandas as pd 
import numpy as np 

from src.components.data.transform.game_base import games_base
from src.components.data.transform.team_form import team_form
from src.components.data.transform.table import table
from src.components.data.transform.player import player_data

def tbl_interactions_features(df, features, interaction_with, prefix):
    for f in features:
        df[prefix + f] = (df[interaction_with] * df[f]) * (df[interaction_with] / 38)
    return df

def create_data():

    print("Create_data - Games base")
    data = games_base()

    print("Create_data - Team form")
    form = team_form
    home_team_form_3, away_team_form_3 = form.overall_form(last_n_games=3)
    home_team_home_form_3 = form.home_away_form(last_n_games=3, home_team=1)
    away_team_away_form_3 = form.home_away_form(last_n_games=3, home_team=0)

    home_team_form_5, away_team_form_5 = form.overall_form(last_n_games=5)
    home_team_home_form_5 = form.home_away_form(last_n_games=5, home_team=1)
    away_team_away_form_5 = form.home_away_form(last_n_games=5, home_team=0)

    home_team_form_10, away_team_form_10 = form.overall_form(last_n_games=10)
    home_team_home_form_10 = form.home_away_form(last_n_games=10, home_team=1)
    away_team_away_form_10 = form.home_away_form(last_n_games=10, home_team=0)

    print("Create_data - Table")
    table_features = table()

    print("Create_data - Player")
    player = player_data()

    print("Create_data - Merging and interactions")
    # merge form
    #data = data.merge(home_team_form_3, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    #data = data.merge(away_team_form_3, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    #data = data.merge(home_team_home_form_3, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_home'], how='inner').drop(['next_id_home', 'team_id_season'], axis=1)
    #data = data.merge(away_team_away_form_3, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_away'], how='inner').drop(['next_id_away', 'team_id_season'], axis=1)
    
    data = data.merge(home_team_form_5, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(away_team_form_5, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(home_team_home_form_5, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_home'], how='inner').drop(['next_id_home', 'team_id_season'], axis=1)
    data = data.merge(away_team_away_form_5, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_away'], how='inner').drop(['next_id_away', 'team_id_season'], axis=1)
    
    #data = data.merge(home_team_form_10, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    #data = data.merge(away_team_form_10, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    #data = data.merge(home_team_home_form_10, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_home'], how='inner').drop(['next_id_home', 'team_id_season'], axis=1)
    #data = data.merge(away_team_away_form_10, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_away'], how='inner').drop(['next_id_away', 'team_id_season'], axis=1)

    # merge table
    data = data.merge(table_features.add_prefix('tbl_home_'), left_on=['season_start_year', 'kickoff_date', 'team_h'], right_on=['tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season'], how='inner')
    data = data.merge(table_features.add_prefix('tbl_away_'), left_on=['season_start_year', 'kickoff_date', 'team_a'], right_on=['tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'], how='inner')
    data = data.drop(['tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season', 'tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'],axis=1)
    
    # table interactions
    tbl_home_interaction_features = ["tbl_home_points_to_team_above", "tbl_home_points_to_team_below", "tbl_home_points_to_win", "tbl_home_points_to_cl", "tbl_home_points_to_euro", "tbl_home_points_to_regulation"]
    tbl_away_interaction_features = ["tbl_away_points_to_team_above", "tbl_away_points_to_team_below", "tbl_away_points_to_win", "tbl_away_points_to_cl", "tbl_away_points_to_euro", "tbl_away_points_to_regulation"]

    data = tbl_interactions_features(df = data, features = tbl_home_interaction_features, interaction_with='tbl_home_number_of_games', prefix='tbl_home_nog_')
    data = tbl_interactions_features(df = data, features = tbl_away_interaction_features, interaction_with='tbl_away_number_of_games', prefix='tbl_away_nog_')
 
    # merge player
    data = data.merge(player.add_prefix('player_home_'), left_on=['season_start_year', 'id', 'team_h'], right_on=['player_home_season_start_year', 'player_home_next_id', 'player_home_team_id_season'], how='left')
    data = data.merge(player.add_prefix('player_away_'), left_on=['season_start_year', 'id', 'team_a'], right_on=['player_away_season_start_year', 'player_away_next_id', 'player_away_team_id_season'], how='left')
    data = data.drop(['player_home_season_start_year', 'player_away_season_start_year', 'player_home_next_id', 'player_away_next_id', 'player_home_team_id_season', 'player_away_team_id_season'], axis=1)

    # player interactions
    player_home_features_terms = ['player_home_max_all_', 'player_home_max_g_', 'player_home_max_d_', 'player_home_max_m_', 'player_home_max_f_']
    player_away_features_terms = [] 
    target_char = "_home_"
    replacement_char = "_away_"

    for i in range(len(player_home_features_terms)):
        player_away_features_terms.append(player_home_features_terms[i].replace(target_char, replacement_char))

    player_home_features = [element for element in data.columns if any(term in element for term in player_home_features_terms)]
    player_away_features = [element for element in data.columns if any(term in element for term in player_away_features_terms)]

    player_diff_features = []
    target_char = "_home_"
    replacement_char = "_diff_"

    for i in range(len(player_home_features)):
        player_diff_features.append(player_home_features[i].replace(target_char, replacement_char))

    # Joining columns using pd.concat(axis=1)
    diff_columns = [diff for diff in player_diff_features]
    diff_values = data[player_home_features].values - data[player_away_features].values
    diff_df = pd.DataFrame(diff_values, columns=diff_columns)
    data = pd.concat([data, diff_df], axis=1)

    # Create a new copy of the DataFrame
    #new_data = data.copy()

    # write
    data.to_csv('artifacts/data.csv', index=False)
    print("Create_data - DONE! \n")