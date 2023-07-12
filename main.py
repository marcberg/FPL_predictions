import os
print(os.getcwd())

# API-functions
from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info

# Fetch data-functions
from src.components.data.fetch_data import fetch_data

fetch_data(get_game_list)
fetch_data(get_player_details)
fetch_data(get_player_hist, season_specific=False)
fetch_data(get_player_info)
fetch_data(get_player_id, season_specific=False)
fetch_data(get_player_name, season_specific=False)
fetch_data(get_round_info)

# Transform data
from src.components.data.transform_data import create_data

create_data()

# ML
from src.components.ml.data_ingest_transform_train import DataIngest, DataTranformTrain

data_ingest = DataIngest()  
data_ingest.create_train_and_test()  

algo_1 = DataTranformTrain(label = 'label_1', drop_labels_list = ['label_1', 'label_X', 'label_2'])
algo_best_model_metric_1, algo_best_param_1, algo_best_model_1 = algo_1.grid_search()

algo_X = DataTranformTrain(label = 'label_X', drop_labels_list = ['label_1', 'label_X', 'label_2'])
algo_best_model_metric_X, algo_best_param_X, algo_best_model_X = algo_X.grid_search()

algo_2 = DataTranformTrain(label = 'label_2', drop_labels_list = ['label_1', 'label_X', 'label_2'])
algo_best_model_metric_2, algo_best_param_2, algo_best_model_2 = algo_2.grid_search()

# Score
from src.components.ml.score import extract_best_model, predict_result

best_model_1 = extract_best_model(algo_best_model_1, algo_best_model_metric_1)
best_model_X = extract_best_model(algo_best_model_X, algo_best_model_metric_X)
best_model_2 = extract_best_model(algo_best_model_2, algo_best_model_metric_2)

predictions = predict_result(best_model_1, best_model_X, best_model_2)
predictions.to_csv('artifacts/result_predictions.csv',index=False,header=True)