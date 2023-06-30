import os
print(os.getcwd())

from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info

from src.components.data.fetch_data import fetch_data

fetch_data(get_game_list)
fetch_data(get_player_details)
fetch_data(get_player_hist, season_specific=False)
fetch_data(get_player_info)
fetch_data(get_player_id, season_specific=False)
fetch_data(get_player_name, season_specific=False)
fetch_data(get_round_info)