import pandas as pd
import numpy as np

from game_functions import get_game_list
from player_functions import get_player_details, get_player_info

def create_train_and_score_df(seasons=[20, 21, 22], score_seasons=[False, False, True]):
    n_seasons = len(seasons)
    list_df = []

    for i in range(n_seasons):
        season = seasons[i]
        score_season = score_seasons[i]

        if not score_season:
            game_list = get_game_list(season=season)
        else:
            game_list = get_game_list()

        team_id = pd.concat([
            game_list[['team_h', 'home']].rename(columns={'team_h': 'team', 'home': 'team_name'}),
            game_list[['team_a', 'away']].rename(columns={'team_a': 'team', 'away': 'team_name'})
        ]).drop_duplicates()

        train_games = game_list.loc[game_list['team_h_score'].notna(), ['id', 'team_h', 'team_a', 'team_h_score', 'team_a_score', 'kickoff']]
        train_games['result'] = train_games.apply(lambda row: '1' if row['team_h_score'] > row['team_a_score'] else '2' if row['team_h_score'] < row['team_a_score'] else 'X', axis=1)
        train_games['match_day'] = train_games['kickoff'].str[:10].astype('datetime64[ns]')
        train_games = train_games.drop(columns=['kickoff', 'team_h_score', 'team_a_score'])
    
        if score_season:
            score_games = game_list.loc[game_list['team_h_score'].isna(), ['id', 'team_h', 'team_a', 'kickoff']]
            score_games['result'] = pd.NA
            score_games['match_day'] = pd.to_datetime(score_games['kickoff'].str[:10])
            score_games = score_games.drop(columns=['kickoff'])

            teams_next_game = pd.concat([
                score_games[['team_h', 'id', 'match_day']].rename(columns={'team_h': 'team'}),
                score_games[['team_a', 'id', 'match_day']].rename(columns={'team_a': 'team'})
            ]).rename(columns={'team_h': 'team'}).drop_duplicates()
            teams_next_game['hierarchy'] = teams_next_game.groupby('team')['match_day'].rank(method='first')
            teams_next_game = teams_next_game.loc[teams_next_game['hierarchy'] == 1].drop(columns=['hierarchy'])

            next_fixture = pd.concat([
                score_games[['team_h', 'id', 'match_day']].rename(columns={'team_h': 'team'}),
                score_games[['team_a', 'id', 'match_day']].rename(columns={'team_a': 'team'}),
                train_games[['team_h', 'id', 'match_day']].rename(columns={'team_h': 'team'}),
                train_games[['team_a', 'id', 'match_day']].rename(columns={'team_a': 'team'})
            ]).sort_values(by=['team', 'match_day'])
            next_fixture['next_fixture'] = next_fixture.groupby('team')['id'].shift(-1)
            next_fixture = next_fixture.drop(columns=['match_day']).reset_index(drop=True)

        else:
            next_fixture = pd.concat([
                train_games[['team_h', 'id', 'match_day']].rename(columns={'team_h': 'team'}),
                train_games[['team_a', 'id', 'match_day']].rename(columns={'team_a': 'team'})
            ]).sort_values(by=['team', 'match_day'])
            next_fixture['next_fixture'] = next_fixture.groupby('team')['id'].shift(-1)
            next_fixture = next_fixture.drop(columns=['match_day']).reset_index(drop=True)

        next_fixture = next_fixture.astype({'next_fixture': pd.Int64Dtype()}).reset_index(drop=True)

        player_data = get_player_details(season=season) if not score_season else get_player_details()
        player_info = get_player_info(season=season) if not score_season else get_player_info()

    

        player = player_data.copy()
        player['match_day'] = pd.to_datetime(player['kickoff_time'].str[:10])
        player = player.rename(columns={'element': 'player_id'})
        player = player.drop(columns=['kickoff_time'])
        player = player.merge(game_list[['id', 'team_h', 'team_a']], left_on='fixture', right_on='id', how='left')
        player['team'] = player.apply(lambda row: row['team_h'] if row['was_home'] else row['team_a'], axis=1)
        player = player.drop(columns=['team_h', 'team_a'])
        player = player.groupby('player_id').apply(lambda x: x.sort_values(by='fixture')).reset_index(drop=True)
        player['goals_per_hour'] = (player['goals_scored'] / player['minutes']) * 60
        player['assists_per_hour'] = (player['assists'] / player['minutes']) * 60
        player['goals_conceded_per_hour'] = (player['goals_conceded'] / player['minutes']) * 60
        player['temp'] = player['minutes'].apply(lambda x: 0 if x < 45 else 1)

        player['total_goals'] = player.groupby('player_id')['goals_scored'].rolling(window=38, min_periods=1).sum().reset_index(level=0, drop=True)
        player['total_assists'] = player.groupby('player_id')['assists'].rolling(window=38, min_periods=1).sum().reset_index(level=0, drop=True)
        player['mean_goal_5'] = player.groupby('player_id')['goals_scored'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_goal_10'] = player.groupby('player_id')['goals_scored'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_goal_per_hour_5'] = player.groupby('player_id')['goals_per_hour'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_goal_per_hour_10'] = player.groupby('player_id')['goals_per_hour'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_assists_5'] = player.groupby('player_id')['assists'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_assists_10'] = player.groupby('player_id')['assists'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_assists_per_hour_5'] = player.groupby('player_id')['assists_per_hour'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_assists_per_hour_10'] = player.groupby('player_id')['assists_per_hour'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_goals_conceded_5'] = player.groupby('player_id')['goals_conceded'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
        player['mean_goals_conceded_per_hour_5'] = player.groupby('player_id')['goals_conceded_per_hour'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        player['mean_goals_conceded_per_hour_10'] = player.groupby('player_id')['goals_conceded_per_hour'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)

        player['n_games'] = player.groupby('player_id').cumcount() + 1
        player['n_games'] = np.where(player['minutes'] > 45, player['n_games'], player['n_games'] - 1)

        player['minutes_5'] = player.groupby('player_id')['minutes'].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)

        player = player.merge(player_info[['id', 'element_type']], left_on='player_id', right_on='id', how='left')
        player = player.drop(columns=['id_y'])
        player['element_type'] = player['element_type'].map({1: 'G', 2: 'D', 3: 'M', 4: 'F'})
        player = player.rename(columns={'element_type': 'position'})
        player = player.merge(next_fixture, left_on=['team', 'fixture'], right_on=['team', 'id'], how='left')
        player = player.sort_values(by=['player_id', 'fixture'])

        player_didnt_play = player[(player['minutes'] == 0) | (player['minutes'].isna())]
        player_didnt_play = player_didnt_play[['team', 'fixture', 'player_id', 'position']]
        player_didnt_play = player_didnt_play.sort_values(by=['team', 'fixture', 'player_id'])

        if score_season:
            player_will_not_play = player_info[player_info['chance_of_playing_this_round'] <= 25]
            player_will_not_play['element_type2'] = np.where(player_will_not_play['element_type'] == 1, 1, 2)
            player_will_not_play['position'] = np.select(
                [
                    player_will_not_play['element_type'] == 1,
                    player_will_not_play['element_type'] == 2,
                    player_will_not_play['element_type'] == 3,
                    player_will_not_play['element_type'] == 4
                ],
                ['G', 'D', 'M', 'F'],
                default=None
            )
            player_will_not_play = player_will_not_play.merge(
                teams_next_game.drop(columns='match_day').rename(columns={'id': 'fixture'}),
                left_on='team',
                right_on='team',
                how='left'
            )
            player_will_not_play = player_will_not_play.rename(columns={'id': 'player_id'})
            player_will_not_play = player_will_not_play[['team', 'fixture', 'player_id', 'position']]
            player_will_not_play = player_will_not_play.merge(
                train_games[['id']].drop_duplicates(),
                left_on='fixture',
                right_on='id',
                how='left'
            )
            player_will_not_play = player_will_not_play[player_will_not_play['id'].isna()]
            player_will_not_play = player_will_not_play.drop(columns=['id'])
            player_will_not_play = player_will_not_play.groupby(['team', 'player_id']).apply(lambda x: x[x['fixture'] == x['fixture'].min()])
            player_will_not_play = player_will_not_play.reset_index(drop=True).drop(columns='fixture')

            team_top = player[~player['team'].isna()].groupby(['team', 'fixture', 'next_fixture'], as_index=False).agg(
                n=('team', 'size'),
                max_value=('value', 'max'),
                max_goal=('total_goals', 'max'),
                max_assists=('total_assists', 'max'),
            ).merge(
                player[(~player['team'].isna()) & (player['position'] == 'G')].groupby(['team', 'fixture', 'next_fixture'], as_index=False).agg(
                        max_value_position_g=('value', 'max'),
                ),
                on=['team', 'fixture', 'next_fixture'],
                how='left'
            ).merge(
                player[(~player['team'].isna()) & (player['position'] == 'D')].groupby(['team', 'fixture', 'next_fixture'], as_index=False).agg(
                        max_value_position_d=('value', 'max'),
                        max_assists_position_d=('total_assists', 'max'),
                ),
                on=['team', 'fixture', 'next_fixture'],
                how='left'
            ).merge(
                player[(~player['team'].isna()) & (player['position'] == 'M')].groupby(['team', 'fixture', 'next_fixture'], as_index=False).agg(
                        max_value_position_m=('value', 'max'),
                        max_goal_position_m=('total_goals', 'max'),
                        max_assists_position_m=('total_assists', 'max'),
                ),
                on=['team', 'fixture', 'next_fixture'],
                how='left'
            ).merge(
                player[(~player['team'].isna()) & (player['position'] == 'F')].groupby(['team', 'fixture', 'next_fixture'], as_index=False).agg(
                        max_value_position_f=('value', 'max'),
                        max_goal_position_f=('total_goals', 'max'),
                        max_assists_position_f=('total_assists', 'max'),
                ),
                on=['team', 'fixture', 'next_fixture'],
                how='left'
            ).sort_values(['team', 'fixture', 'next_fixture'])

    return player_will_not_play

#if __name__ == '__main__':
a = create_train_and_score_df(seasons=[22], score_seasons=[True])

print(a)