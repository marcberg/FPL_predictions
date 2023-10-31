import os
import numpy as np 
import pandas as pd 
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb 
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

from src.components.data.api.game_functions import get_game_list
from src.components.data.api.player_functions import get_player_details, get_player_info, get_player_hist, get_player_id, get_player_name
from src.components.data.api.round_functions import get_round_info
from src.components.data.fetch_data import fetch_data
from src.components.data.transform_data import create_data
from src.components.ml.data_ingest_transform_train import DataIngest, DataTranformTrain
from src.components.ml.score import predict_result

# API-functions
def get_data():

    fetch_data(get_game_list, id_list = ["id","team_h","team_a","season_start_year"])
    fetch_data(get_player_details, id_list = ["season_start_year","element","fixture"])
    fetch_data(get_player_hist, id_list = ["season_name","element_code"], season_specific=False)
    fetch_data(get_player_info, id_list = ["season_start_year", "id"])
    fetch_data(get_player_id, id_list = ["id"], season_specific=False)
    fetch_data(get_player_name, id_list = ["id"], season_specific=False)
    fetch_data(get_round_info, id_list = ["id", "season_start_year"])

# Transform data
def transform_data():

    create_data()

# ML
def setup_train_models():

    data_ingest = DataIngest()  
    data_ingest.create_train_and_test()  

def models_params():
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": xgb.XGBClassifier(),
        "LightGBM": lgb.LGBMClassifier(),
    }

    params = {
        "Logistic Regression":{
            'model__C': [0.001, 0.01, 0.1, 1, 10], 
            'model__penalty': ['l1', 'l2'],  
            'model__max_iter': [1000, 10000],  
            'model__solver': ['liblinear', 'saga']  
        },
        "Decision Tree": {
            'model__criterion': ['entropy', 'gini'], 
            'model__max_depth': [None, 2, 3, 4, 5, 7, 9, 12, 15], 
            'model__min_samples_leaf': [1, 2, 5, 10, 20],  
            'model__min_samples_split': [2, 5, 10],  
        },
        "Random Forest":{
            'model__bootstrap': [True],
            'model__max_features': ['sqrt', 'log2', None],
            #'model__max_features': [10, 20, 50],
            'model__max_depth': [2, 3, 4, 5, 7, 9, 12, 15],
            'model__min_samples_leaf': [1, 2, 4, 5, 10, 20, 50],
            'model__n_estimators': [10, 50, 100, 500, 1000],
        },
        "Gradient Boosting":{
            "model__loss":["log_loss", "exponential"],
            'model__learning_rate': [0.001, 0.005, 0.01, 0.015, 0.03, 0.06],
            'model__min_samples_leaf': [1, 2, 5, 10, 20, 50],
            'model__max_depth': [2, 3, 4, 5, 7, 9, 11, 15],
            'model__n_estimators': [10, 50, 100],
        },
        "XGBoost":{
            'model__max_depth': [2, 3, 4, 5, 7, 9, 12, 15],
            'model__learning_rate': [0.001, 0.005, 0.01, 0.015, 0.03],
            'model__n_estimators': [10, 50, 100],
            'model__min_child_weight': [5, 10, 50],
            'model__gamma': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1]
        }, 
        "LightGBM":{
            'model__max_depth': [2, 3, 4, 5, 7, 9, 12, 15],
            'model__learning_rate': [0.001, 0.005, 0.01, 0.015, 0.03],
            'model__num_leaves': [31,63,127,255, 511],
            'model__min_child_samples': [10, 20, 30],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
            'model__reg_alpha': [0.0, 0.1],
            'model__reg_lambda': [0.0, 0.1],
            'model__n_estimators':[100, 200, 300, 500]
        }
    }

    return models, params


def train(label, save_to_mlflow=True, random_grid=True, n_random_hyperparameters=30):

    models, params = models_params()

    algo_1 = DataTranformTrain(label = label)
    algo_1.grid_search(models=models, params=params, save_to_mlflow=save_to_mlflow, random_grid=random_grid, n_random_hyperparameters=n_random_hyperparameters)

# Select algo that performs best overall 
def score():
    all_algo_metrics_1 = pd.read_excel("artifacts/ml_results/label_1/all_algo_metrics.xlsx")
    all_algo_metrics_X = pd.read_excel("artifacts/ml_results/label_X/all_algo_metrics.xlsx")
    all_algo_metrics_2 = pd.read_excel("artifacts/ml_results/label_2/all_algo_metrics.xlsx")

    concat_matrics = pd.concat([all_algo_metrics_1, all_algo_metrics_X, all_algo_metrics_2])
    calculate_total_metrics = concat_matrics.groupby("Algorithm", as_index=False)["AUC-ROC Val"].sum()
    rearrange_metrics = calculate_total_metrics.sort_values("AUC-ROC Val", ascending=False)["Algorithm"].reset_index(drop=True)
    best_total_algorithm = rearrange_metrics[0]
    print("\nBest overall algorithm: ", best_total_algorithm, "\n")

    best_total_algorithm_1 = joblib.load('artifacts/ml_results/label_1/{0}.pkl'.format(best_total_algorithm))
    best_total_algorithm_X = joblib.load('artifacts/ml_results/label_X/{0}.pkl'.format(best_total_algorithm))
    best_total_algorithm_2 = joblib.load('artifacts/ml_results/label_2/{0}.pkl'.format(best_total_algorithm))

    predictions = predict_result(best_total_algorithm_1, best_total_algorithm_X, best_total_algorithm_2, predict_data='score')

    print(predictions.to_string(index=False))

    # Keep historical data in case it is missing in new download
    try:
        prev_data = pd.read_csv('artifacts/result_predictions.csv')
        
        predictions = pd.concat([prev_data, predictions], ignore_index=True).drop_duplicates(subset=['kickoff_date', 'home', 'away'], keep="last")

    except FileNotFoundError:
        pass

    predictions.to_csv('artifacts/result_predictions.csv',index=False,header=True)

    

