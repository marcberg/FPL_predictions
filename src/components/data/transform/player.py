import pandas as pd
import numpy as np

from src.components.data.transform.team_form import team_form

def hourly_rate(df,features):
    for f in features:
        df["hourly_rate_"+f] = np.where(df['minutes'] > 0, (df[f] / df['minutes'])*60, 0)
    return df

def player_impact_per_game(df,features):
    team_total_minutes = df.groupby(["season_start_year", "team_id_season", "fixture"], as_index=False, dropna=False).agg(team_total_minutes = ('minutes', np.sum))
    df = df.merge(team_total_minutes, on=["season_start_year", "team_id_season", "fixture"], how="left")
    df['impact_on_game_by_minutes'] = df['minutes'] / df['team_total_minutes']

    for f in features:
        df["impact_game_"+f] = df[f] * df['impact_on_game_by_minutes']

    return df

def player_rolling_last_x_games(df, features_to_mean, features_to_max, x_games=6):
    df = df.sort_values(["season_start_year", "player_id", "kickoff_date"]).reset_index(drop=True)
    
    for f in features_to_mean:
        df['rolling_mean_'+str(x_games)+'_'+f] = df.groupby(['season_start_year','player_id'])[f].rolling(window=x_games, min_periods=1).mean().reset_index(drop=True)
    
    for f in features_to_max:
        df['rolling_max_'+str(x_games)+'_'+f] = df.groupby(['season_start_year','player_id'])[f].rolling(window=x_games, min_periods=1).max().reset_index(drop=True)

    return df

def player_data():
    player_details = pd.read_csv('artifacts/fetched_data/get_player_details.csv')
    player_info = pd.read_csv('artifacts/fetched_data/get_player_info.csv')

    # fix with data
    player_details = player_details.rename(columns={"element":"player_id"})
    player_details['kickoff_date'] = pd.to_datetime(player_details['kickoff_time'])
    player_details['kickoff_date'] = player_details['kickoff_date'].dt.strftime('%Y-%m-%d')

    # add position
    player_info = player_info[["season_start_year", "id", "element_type"]].rename(columns={"id":"player_id"})
    player = player_details.merge(player_info, on=["season_start_year", "player_id"])
    player['position'] = np.where(player.element_type == 1, "G", np.where(player.element_type == 2, "D", np.where(player.element_type == 3, "M", "F")))

    # add team
    game_list = pd.read_csv('artifacts/fetched_data/get_game_list.csv')
    game_list = game_list[["season_start_year", "id", "team_h", "team_a"]].rename(columns={"id":"fixture"})
    player = player.merge(game_list, on=["season_start_year", "fixture"])
    player['team_id_season'] = np.where(player.was_home == True, player.team_h, player.team_a)
    player = player.drop(["team_h", "team_a"], axis=1)

    # add next_fixture
    teams_fixture = team_form.data_setup()[["season_start_year", "team_id_season", "id", "next_id"]].rename(columns={"id":"fixture"}) # problem when player change team, then next-fixture will be the new team next fixture
    player = player.merge(teams_fixture, on=["season_start_year", "team_id_season", "fixture"], how="inner")

    # sorting
    player = player.sort_values(["season_start_year", "player_id", "kickoff_date"]).reset_index(drop=True)
    player = player.drop(["kickoff_time", "own_goals", "penalties_saved", "penalties_missed", "bonus", "bps", "ict_index", "element_type", "opponent_team"], axis=1)

    to_hourly_rate_features = ["goals_scored","assists","clean_sheets","goals_conceded","saves"]
    player = hourly_rate(player,to_hourly_rate_features)

    to_impact_features = ["influence","creativity","threat","value","transfers_balance","selected"]
    hourly_rate_features = [value for value in player.columns.to_list() if "hourly_rate" in value]
    player = player_impact_per_game(player, to_impact_features + hourly_rate_features)

    impact_game_features = [value for value in player.columns.to_list() if "impact_game" in value]

    features_to_mean = hourly_rate_features + impact_game_features
    features_to_max = impact_game_features
    player = player_rolling_last_x_games(player, features_to_mean, features_to_max)

    # agg to team
    rolling_features = [value for value in player.columns.to_list() if "rolling_" in value]

    max_all = player.groupby(['season_start_year', 'team_id_season', 'next_id']).apply(lambda x: pd.Series({
        f'max_all_{col}': x[col].max()
        for col in rolling_features
    })).reset_index()

    desired_terms = ['saves', 'clean_sheet', 'value', 'selected', 'goals_conceded']
    g_features = [element for element in rolling_features if any(term in element for term in desired_terms)]

    max_g = player.loc[player.position == "G"].groupby(['season_start_year', 'team_id_season', 'next_id']).apply(lambda x: pd.Series({
        f'max_g_{col}': x[col].max()
        for col in g_features
    })).reset_index()

    not_desired_terms = ['saves', 'clean_sheet']
    d_m_f_features = [element for element in rolling_features if not any(term in element for term in desired_terms)]

    max_d = player.loc[player.position == "D"].groupby(['season_start_year', 'team_id_season', 'next_id']).apply(lambda x: pd.Series({
        f'max_d_{col}': x[col].max()
        for col in d_m_f_features
    })).reset_index()

    max_m = player.loc[player.position == "M"].groupby(['season_start_year', 'team_id_season', 'next_id']).apply(lambda x: pd.Series({
        f'max_m_{col}': x[col].max()
        for col in d_m_f_features
    })).reset_index()

    max_f = player.loc[player.position == "F"].groupby(['season_start_year', 'team_id_season', 'next_id']).apply(lambda x: pd.Series({
        f'max_f_{col}': x[col].max()
        for col in d_m_f_features
    })).reset_index()

    team_player_stats = max_all.merge(max_g, on=['season_start_year', 'team_id_season', 'next_id'], how="left")
    team_player_stats = team_player_stats.merge(max_d, on=['season_start_year', 'team_id_season', 'next_id'], how="left")
    team_player_stats = team_player_stats.merge(max_m, on=['season_start_year', 'team_id_season', 'next_id'], how="left")
    team_player_stats = team_player_stats.merge(max_f, on=['season_start_year', 'team_id_season', 'next_id'], how="left")

    return team_player_stats