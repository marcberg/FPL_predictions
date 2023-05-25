import pandas as pd 
import os

games = pd.read_csv(os.getcwd() + "/../../../../artifacts/fetched_data/get_game_list.csv")
print(games)
