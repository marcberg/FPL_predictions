import pandas as pd 
import numpy as np
import os

class team_form():

    def data_setup():
        games = pd.read_csv(os.getcwd() + "/artifacts/fetched_data/get_game_list.csv")

        home = games.rename(columns = {'home':'team', 'team_h':'team_id_season'}).drop(['away', 'team_a'], axis=1)
        home['home'] = 1

        away = games.rename(columns = {'away':'team', 'team_a':'team_id_season'}).drop(['home', 'team_h'], axis=1)
        away['home'] = 0

        team_games = pd.concat([home, away])

        team_games['win'] = np.where((team_games.home == 1) & (team_games['team_h_score'] > team_games['team_a_score']), 1, 
                                    np.where((team_games.home == 0) & (team_games['team_h_score'] < team_games['team_a_score']), 1, 0))
        team_games['draw'] = np.where((team_games['team_h_score'] == team_games['team_a_score']), 1, 0)
        team_games['loss'] = np.where((team_games.home == 1) & (team_games['team_h_score'] < team_games['team_a_score']), 1, 
                                    np.where((team_games.home == 0) & (team_games['team_h_score'] > team_games['team_a_score']), 1, 0))

        team_games['goals_scored'] = np.where(team_games.home == 1, team_games['team_h_score'], team_games['team_a_score'])
        team_games['goals_conceded'] = np.where(team_games.home == 1, team_games['team_a_score'], team_games['team_h_score'])

        team_games['kickoff'] = pd.to_datetime(team_games['kickoff'])
        team_games = team_games.sort_values('kickoff').reset_index(drop=True)

        team_games['next_id'] = team_games.groupby(['season_start_year', 'team_id_season'])['id'].shift(-1)

        team_games = team_games.drop(['team_h_score', 'team_a_score'], axis=1)

        return team_games
    
    team_games = data_setup()
    
    def overall_form(last_n_games, team_games=team_games):
        '''
        
        '''
        overall_form = team_games.sort_values(['season_start_year', 'team_id_season', 'kickoff']).reset_index(drop=True)

        overall_form['win_share_latest_{0}_games'.format(last_n_games)] = overall_form.groupby(['season_start_year', 'team_id_season'])['win'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        overall_form['draw_share_latest_{0}_games'.format(last_n_games)] = overall_form.groupby(['season_start_year', 'team_id_season'])['draw'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        overall_form['loss_share_latest_{0}_games'.format(last_n_games)] = overall_form.groupby(['season_start_year', 'team_id_season'])['loss'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)

        overall_form['avg_goals_scored_latest_{0}_games'.format(last_n_games)] = overall_form.groupby(['season_start_year', 'team_id_season'])['goals_scored'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        overall_form['avg_goals_conceded_latest_{0}_games'.format(last_n_games)] = overall_form.groupby(['season_start_year', 'team_id_season'])['goals_conceded'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)

        home_team_form = overall_form.rename(columns={
                                'win_share_latest_{0}_games'.format(last_n_games):'win_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                'draw_share_latest_{0}_games'.format(last_n_games):'draw_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                'loss_share_latest_{0}_games'.format(last_n_games):'loss_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                'avg_goals_scored_latest_{0}_games'.format(last_n_games):'avg_goals_scored_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                'avg_goals_conceded_latest_{0}_games'.format(last_n_games):'avg_goals_conceded_latest_{0}_games_overall_home_team'.format(last_n_games)})
        home_team_form = home_team_form[['team_id_season', 
                                            'next_id', 
                                            'season_start_year', 
                                            'win_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                            'draw_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                            'loss_share_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                            'avg_goals_scored_latest_{0}_games_overall_home_team'.format(last_n_games), 
                                            'avg_goals_conceded_latest_{0}_games_overall_home_team'.format(last_n_games)]]

        away_team_form = overall_form.rename(columns={
                                'win_share_latest_{0}_games'.format(last_n_games):'win_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                'draw_share_latest_{0}_games'.format(last_n_games):'draw_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                'loss_share_latest_{0}_games'.format(last_n_games):'loss_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                'avg_goals_scored_latest_{0}_games'.format(last_n_games):'avg_goals_scored_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                'avg_goals_conceded_latest_{0}_games'.format(last_n_games):'avg_goals_conceded_latest_{0}_games_overall_away_team'.format(last_n_games)})
        away_team_form = away_team_form[['team_id_season', 
                                            'next_id', 
                                            'season_start_year', 
                                            'win_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                            'draw_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                            'loss_share_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                            'avg_goals_scored_latest_{0}_games_overall_away_team'.format(last_n_games), 
                                            'avg_goals_conceded_latest_{0}_games_overall_away_team'.format(last_n_games)]]
        
        return home_team_form, away_team_form
    
    def home_away_form(home_team, last_n_games, team_games=team_games):
        '''
        
        '''

        if home_team == 1:
            team = "home"
        else:
            team = "away"
        
        games = team_games.loc[team_games.home == home_team].sort_values(['season_start_year', 'team_id_season', 'kickoff']).reset_index(drop=True)
        games['next_id_{0}'.format(team)] = games.groupby('team_id_season')['id'].shift(-1)

        games['win_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)] = games.groupby(['season_start_year', 'team_id_season'])['win'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        games['draw_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)] = games.groupby(['season_start_year', 'team_id_season'])['draw'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        games['loss_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)] = games.groupby(['season_start_year', 'team_id_season'])['loss'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)

        games['avg_goals_scored_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)] = games.groupby(['season_start_year', 'team_id_season'])['goals_scored'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)
        games['avg_goals_conceded_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)] = games.groupby(['season_start_year', 'team_id_season'])['goals_conceded'].rolling(window=last_n_games, min_periods=1).mean().reset_index(drop=True)

        games = games[['team_id_season', 
                                'next_id_{0}'.format(team), 
                                'season_start_year', 
                                'win_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games), 
                                'draw_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games), 
                                'loss_share_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games), 
                                'avg_goals_scored_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games), 
                                'avg_goals_conceded_latest_{1}_games_{0}_{0}_team'.format(team, last_n_games)]]

        return games

