import pandas as pd

from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info

game_list = get_game_list()

converted_timestamps = pd.to_datetime(game_list.kickoff)

fetch_current_season = min(converted_timestamps.dt.strftime('%y').astype(int))
fetch_previous_seasons = range(fetch_current_season-4, fetch_current_season-1)

def fetch_data(fetching_function, current_season=fetch_current_season, previous_seasons=fetch_previous_seasons, season_specific=True):

    season_data = fetching_function()

    if season_specific:
        season_data['season_start_year'] = int(current_season)

        for i in previous_seasons:
            season_data_i = fetching_function(season=i).drop(['Unnamed: 0'], axis=1)
            season_data_i['season_start_year'] = int(i)
            season_data = pd.concat([season_data, season_data_i])
        
    season_data.to_csv('artifacts/fetched_data/' + fetching_function.__name__ + '.csv', index=False)
    print('data/' + fetching_function.__name__ + '.csv is fetched.')

if __name__=="__main__":
    fetch_data(get_game_list)
    fetch_data(get_player_details)
    fetch_data(get_player_hist, season_specific=False)
    fetch_data(get_player_info)
    fetch_data(get_player_id, season_specific=False)
    fetch_data(get_player_name, season_specific=False)
    fetch_data(get_round_info)