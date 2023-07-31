import os
import numpy as np 
import pandas as pd 
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

# API-functions
from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info

# Fetch data-functions
from src.components.data.fetch_data import fetch_data

fetch_data(get_game_list, id_list = ["id","team_h","team_a","season_start_year"])
fetch_data(get_player_details, id_list = ["season_start_year","element","fixture"])
fetch_data(get_player_hist, id_list = ["season_name","element_code"], season_specific=False)
fetch_data(get_player_info, id_list = ["season_start_year", "id"])
fetch_data(get_player_id, id_list = ["id"], season_specific=False)
fetch_data(get_player_name, id_list = ["id"], season_specific=False)
fetch_data(get_round_info, id_list = ["id", "season_start_year"])

# Transform data
from src.components.data.transform_data import create_data

create_data()

# ML
from src.components.ml.data_ingest_transform_train import DataIngest, DataTranformTrain

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb 

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(),
}

params = {
    "Logistic Regression":{
        'model__C': [0.001, 0.01, 0.1, 1, 10], 
        'model__penalty': ['l1', 'l2'],  
        'model__max_iter': [100, 1000, 10000],  
        'model__solver': ['liblinear', 'saga']  
    },
    "Decision Tree": {
        'model__criterion': ['entropy', 'gini'], 
        'model__max_depth': [None, 2, 3, 4, 5, 6], 
        'model__min_samples_leaf': [1, 2, 5, 10, 20],  
        'model__min_samples_split': [2, 5, 10],  
    },
    "Random Forest":{
        'model__bootstrap': [True],
        'model__max_features': ['sqrt', 'log2', None],
        #'model__max_features': [10, 20, 50],
        'model__max_depth': [2, 3, 4, 6],
        'model__min_samples_leaf': [1, 2, 4, 5, 10, 20, 50],
        'model__n_estimators': [10, 50, 100, 500, 1000],
    },
    "Gradient Boosting":{
        "model__loss":["log_loss", "exponential"],
        'model__learning_rate': [0.001, 0.005, 0.01, 0.015, 0.03, 0.06],
        'model__min_samples_leaf': [1, 2, 5, 10, 20, 50],
        'model__max_depth': [2, 3, 4, 6],
        'model__n_estimators': [10, 50, 100],
    },
    "XGBoost":{
        'model__max_depth': [2, 3, 4, 6],
        'model__learning_rate': [0.001, 0.005, 0.01, 0.015, 0.03, 0.06],
        'model__n_estimators': [10, 50, 100, 500],
        'model__min_child_weight': [3, 5, 10, 50],
        'model__gamma': [0, 0.1, 1, 2],
        'model__reg_lambda': [0, 0.1, 1, 10]
    },   
}

data_ingest = DataIngest()  
data_ingest.create_train_and_test()  

algo_1 = DataTranformTrain(label = 'label_1')
algo_1.grid_search(models=models, params=params, save_to_mlflow=False)

algo_X = DataTranformTrain(label = 'label_X')
algo_X.grid_search(models=models, params=params, save_to_mlflow=False)

algo_2 = DataTranformTrain(label = 'label_2')
algo_2.grid_search(models=models, params=params, save_to_mlflow=False)

# Select algo that performs best overall 
all_algo_metrics_1 = pd.read_excel("artifacts/ml_results/label_1/all_algo_metrics.xlsx")
all_algo_metrics_X = pd.read_excel("artifacts/ml_results/label_X/all_algo_metrics.xlsx")
all_algo_metrics_2 = pd.read_excel("artifacts/ml_results/label_2/all_algo_metrics.xlsx")

concat_matrics = pd.concat([all_algo_metrics_1, all_algo_metrics_X, all_algo_metrics_2])
calculate_total_metrics = concat_matrics.groupby("Algorithm", as_index=False)["AUC-ROC Val"].sum()
rearrange_metrics = calculate_total_metrics.sort_values("AUC-ROC Val", ascending=False)["Algorithm"].reset_index(drop=True)
best_total_algorithm = rearrange_metrics[0]

# Score
from src.components.ml.score import predict_result

best_total_algorithm_1 = joblib.load('artifacts/ml_results/label_1/{0}.pkl'.format(best_total_algorithm))
best_total_algorithm_X = joblib.load('artifacts/ml_results/label_X/{0}.pkl'.format(best_total_algorithm))
best_total_algorithm_2 = joblib.load('artifacts/ml_results/label_2/{0}.pkl'.format(best_total_algorithm))

predictions = predict_result(best_total_algorithm_1, best_total_algorithm_X, best_total_algorithm_2, predict_data='score')
predictions.to_csv('artifacts/result_predictions.csv',index=False,header=True)