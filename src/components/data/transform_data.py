import pandas as pd 
import numpy as np 

#from src.components.data.transform.team_form import team_form
from src.components.data.transform.game_base import games_base
from src.components.data.transform.team_form import team_form

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

    data.to_csv('artifacts/data.csv', index=False)