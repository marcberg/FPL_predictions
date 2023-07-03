import pandas as pd
import numpy as np 

from src.components.data.transform.team_form import team_form

def table():
    team_games = team_form.data_setup()
    team_games = team_games.sort_values(['season_start_year', 'team_id_season', 'kickoff']).reset_index(drop=True)

    team_games['points_from_game'] = np.where(team_games['win'] == 1, 3, np.where(team_games['draw'] == 1, 1, 0))
    team_games['game'] = 1

    table = team_games.sort_values(['season_start_year', 'team_id_season', 'kickoff']).reset_index(drop=True)
    table['kickoff_date'] = table['kickoff'].dt.date
    table['team_points'] = table.groupby(['season_start_year', 'team_id_season'])['points_from_game'].cumsum()
    table['number_of_games'] = table.groupby(['season_start_year', 'team_id_season'])['game'].cumsum()
    table = table[['season_start_year', 'kickoff', 'kickoff_date', 'team', 'team_id_season', 'number_of_games', 'points_from_game', 'team_points']]
    table['games_left_season'] = 38 - table['number_of_games']

    dates = pd.DataFrame(table[["season_start_year", "kickoff_date"]].drop_duplicates()).sort_values("kickoff_date").reset_index(drop=True)
    dates = dates.rename(columns={'kickoff_date': 'next_kickoff_date'})


    a = table.merge(dates, on=["season_start_year"])
    a = a.loc[a.kickoff_date < a.next_kickoff_date]
    a['rn'] = a.groupby(['team', 'next_kickoff_date'])['kickoff_date'].rank(ascending=False, method='first')
    a = a.loc[a.rn == 1].drop("rn", axis=1)
    a['position'] = a.groupby(['season_start_year', 'next_kickoff_date'])['team_points'].rank(ascending=False, method='first')
    a = a.sort_values(["season_start_year", "next_kickoff_date", "position"], ascending=[True, True, False]).reset_index(drop=True)

    a['points_to_team_above'] = (a['team_points'] - a.groupby('next_kickoff_date')['team_points'].shift(-1)).fillna(0)
    a['points_to_team_below'] = (a['team_points'] - a.groupby('next_kickoff_date')['team_points'].shift()).fillna(0)

    a['games_left_diff_above'] = (a['games_left_season'] - a.groupby('next_kickoff_date')['games_left_season'].shift(-1)).fillna(0)
    a['games_left_diff_below'] = (a['games_left_season'] - a.groupby('next_kickoff_date')['games_left_season'].shift()).fillna(0)

    win = a.loc[a['position'] == 1][['next_kickoff_date', 'team_points']].rename(columns={'team_points': 'win_points'})
    champions_league = a.loc[a['position'] == 4][['next_kickoff_date', 'team_points']].rename(columns={'team_points': 'cl_points'})
    euro = a.loc[a['position'] == 7][['next_kickoff_date', 'team_points']].rename(columns={'team_points': 'euro_points'})
    regulation = a.loc[a['position'] == 18][['next_kickoff_date', 'team_points']].rename(columns={'team_points': 'regulation_points'})

    a = a.merge(win, on="next_kickoff_date")
    a = a.merge(champions_league, on="next_kickoff_date")
    a = a.merge(euro, on="next_kickoff_date")
    a = a.merge(regulation, on="next_kickoff_date")

    a['points_to_win'] = a['team_points'] - a['win_points']
    a['points_to_cl'] = a['team_points'] - a['cl_points']
    a['points_to_euro'] = a['team_points'] - a['euro_points']
    a['points_to_regulation'] = a['team_points'] - a['regulation_points']

    a = a.drop(['kickoff', 'kickoff_date', 'team', 'win_points', 'cl_points', 'euro_points', 'regulation_points'], axis=1).rename(columns=({'points_from_game':'points_from_last_game'}))

    return a

if __name__=="__main__":
    t = table()
    print(t)