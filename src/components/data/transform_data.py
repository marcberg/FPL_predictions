import pandas as pd 
import numpy as np 

from src.components.data.transform.game_base import games_base
from src.components.data.transform.team_form import team_form
from src.components.data.transform.table import table


def create_data():

    base = games_base()

    form = team_form
    home_team_form, away_team_form = form.overall_form()
    home_team_home_form = form.home_away_form(home_team=1)
    away_team_away_form = form.home_away_form(home_team=0)

    # merge
    data = base.merge(home_team_form, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(away_team_form, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id'], how='inner').drop(['next_id', 'team_id_season'], axis=1)
    data = data.merge(home_team_home_form, left_on=['season_start_year', 'team_h', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_home'], how='inner').drop(['next_id_home', 'team_id_season'], axis=1)
    data = data.merge(away_team_away_form, left_on=['season_start_year', 'team_a', 'id'], right_on=['season_start_year', 'team_id_season', 'next_id_away'], how='inner').drop(['next_id_away', 'team_id_season'], axis=1)

    table_features = table()

    data = data.merge(table_features.add_prefix('tbl_home_'), left_on=['season_start_year', 'kickoff_date', 'team_h'], right_on=['tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season'])
    data = data.merge(table_features.add_prefix('tbl_away_'), left_on=['season_start_year', 'kickoff_date', 'team_h'], right_on=['tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'])
    data = data.drop(['kickoff_date', 'tbl_home_season_start_year', 'tbl_home_next_kickoff_date', 'tbl_home_team_id_season', 'tbl_away_season_start_year', 'tbl_away_next_kickoff_date', 'tbl_away_team_id_season'],axis=1)

    data.to_csv('artifacts/data.csv', index=False)