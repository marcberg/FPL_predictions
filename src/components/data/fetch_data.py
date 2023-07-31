import pandas as pd

from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info

game_list = get_game_list()
game_list = game_list.loc[~game_list.kickoff.isna()]

converted_timestamps = pd.to_datetime(game_list.kickoff)

fetch_current_season = min(converted_timestamps.dt.strftime('%y').astype(int))
fetch_previous_seasons = range(fetch_current_season-4, fetch_current_season)

def fetch_data(fetching_function, 
                id_list,
                current_season=fetch_current_season, 
                previous_seasons=fetch_previous_seasons, 
                season_specific=True):

    season_data = fetching_function()

    if season_specific:
        season_data['season_start_year'] = int(current_season)

        for i in previous_seasons:
            try:
                season_data_i = fetching_function(season=i).drop(['Unnamed: 0'], axis=1)
                season_data_i['season_start_year'] = int(i)
                season_data = pd.concat([season_data, season_data_i])
            except:
                pass
        
    # Keep historical data in case it is missing in new download
    try:
        prev_data = pd.read_csv('artifacts/fetched_data/' + fetching_function.__name__ + '.csv')

        if fetching_function.__name__ == "get_game_list":
            prev_data = prev_data[prev_data.finished == True]
        
        season_data = pd.concat([prev_data, season_data], ignore_index=True).drop_duplicates(subset=id_list, keep="last")

    except FileNotFoundError:
        pass

    season_data.to_csv('artifacts/fetched_data/' + fetching_function.__name__ + '.csv', index=False)
    print('data/' + fetching_function.__name__ + '.csv is fetched.')

if __name__=="__main__":
    fetch_data(get_game_list, id_list = ["id","team_h","team_a","season_start_year"])
    fetch_data(get_player_details, id_list = ["season_start_year","element","fixture"])
    fetch_data(get_player_hist, id_list = ["season_name","element_code"], season_specific=False)
    fetch_data(get_player_info, id_list = ["season_start_year", "id"])
    fetch_data(get_player_id, id_list = ["id"], season_specific=False)
    fetch_data(get_player_name, id_list = ["id"], season_specific=False)
    fetch_data(get_round_info, id_list = ["id", "season_start_year"])