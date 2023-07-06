import pandas as pd 
import numpy as np 

from src.components.data.transform.game_base import games_base
from src.components.data.transform.team_form import team_form
from src.components.data.transform.table import table

def tbl_interactions_features(df, features, interaction_with, prefix):
    for f in features:
        df[prefix + f] = (df[interaction_with] * df[f]) * (df[interaction_with] / 38)
    return df


def create_data():

    base = games_base()

    form = team_form
    home_team_form, away_team_form = form.overall_form()
    home_team_home_form = form.home_away_form(home_team=1)
    away_team_away_form = form.home_away_form(home_team=0)

    table_features = table()

    # merge
    data = base.merge(home_team_form, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(away_team_form, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(home_team_home_form, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_home'], how='inner').drop(['next_id_home', 'team_id_season'], axis=1)
    data = data.merge(away_team_away_form, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_away'], how='inner').drop(['next_id_away', 'team_id_season'], axis=1)

    data = data.merge(table_features.add_prefix('tbl_home_'), left_on=['season_start_year', 'kickoff_date', 'team_h'], right_on=['tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season'])
    data = data.merge(table_features.add_prefix('tbl_away_'), left_on=['season_start_year', 'kickoff_date', 'team_h'], right_on=['tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'])
    data = data.drop(['kickoff_date', 'tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season', 'tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'],axis=1)

    # interactions
    tbl_home_interaction_features = ["tbl_home_points_to_team_above", "tbl_home_points_to_team_below", "tbl_home_points_to_win", "tbl_home_points_to_cl", "tbl_home_points_to_euro", "tbl_home_points_to_regulation"]
    tbl_away_interaction_features = ["tbl_away_points_to_team_above", "tbl_away_points_to_team_below", "tbl_away_points_to_win", "tbl_away_points_to_cl", "tbl_away_points_to_euro", "tbl_away_points_to_regulation"]

    data = tbl_interactions_features(df = data, features = tbl_home_interaction_features, interaction_with='tbl_home_number_of_games', prefix='tbl_home_nog_')
    data = tbl_interactions_features(df = data, features = tbl_away_interaction_features, interaction_with='tbl_away_number_of_games', prefix='tbl_away_nog_')

    # write
    data.to_csv('artifacts/data.csv', index=False)