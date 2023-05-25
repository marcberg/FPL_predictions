import pandas as pd

from api.game_functions import get_game_list
from api.player_functions import get_player_details, get_player_info

game_list = get_game_list()

converted_timestamps = pd.to_datetime(game_list.kickoff)

fetch_current_season = min(converted_timestamps.dt.strftime('%y').astype(int))
fetch_previous_seasons = range(fetch_current_season-3, fetch_current_season)

def fetch_data(fetching_function, current_season=fetch_current_season, previous_seasons=fetch_previous_seasons):

    season_data = fetching_function()
    season_data['season_start_year'] = int(current_season)

    for i in previous_seasons:
        season_data_i = fetching_function(season=i).drop(['Unnamed: 0'], axis=1)
        season_data_i['season_start_year'] = int(i)
        season_data = pd.concat([season_data, season_data_i])
    
    season_data.to_csv('../../../artifacts/fetched_data/' + fetching_function.__name__ + '.csv', index=False)
    print('data/' + fetching_function.__name__ + '.csv is fetched.')

if __name__=="__main__":
    fetch_data(get_game_list)
    fetch_data(get_player_details)
    fetch_data(get_player_info)