import os
import sys

from dataclasses import dataclass

import numpy as np 
import pandas as pd
import mlflow
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb # xgb.XGBClassifier

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)


from src.components.ml.metrics import evaluate_model_kpi

@dataclass
class DataConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    val_data_path: str=os.path.join('artifacts',"validation.csv")
    score_data_path: str=os.path.join('artifacts',"score.csv")

class DataIngest():

    def __init__(self):        
        self.config=DataConfig()

    def create_train_and_test(self):

        df=pd.read_csv(self.config.raw_data_path)
        df_features = df.drop(["season_start_year","GW","id","team_h","team_a","train_score","home","away","kickoff_year","kickoff_month","kickoff_date"], axis=1)
        #os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        train_set, test_set=train_test_split(df_features.loc[df.train_score == "train"],test_size=0.4,random_state=42)
        test_set, val_set=train_test_split(test_set,test_size=0.5,random_state=42)

        train_set.to_csv(self.config.train_data_path,index=False,header=True)
        test_set.to_csv(self.config.test_data_path,index=False,header=True)
        val_set.to_csv(self.config.val_data_path,index=False,header=True)
        
        score = df.loc[df.train_score == "score"]
        h = score[['id', 'team_h']].rename(columns={"team_h":"team"})
        a = score[['id', 'team_a']].rename(columns={"team_a":"team"})

        h_a = pd.concat([h, a]).sort_values(["team", "id"]).reset_index(drop=True)
        h_a['row_number'] = h_a.groupby('team').cumcount() + 1

        next_games = h_a.loc[(h_a['row_number'] == 1)]['id'].drop_duplicates()
        teams_next_game = score.merge(next_games, on="id", how="inner")
        teams_next_game.to_csv(self.config.score_data_path,index=False,header=True)

        return(
            self.config.train_data_path,
            self.config.test_data_path,
            self.config.val_data_path,
            self.config.score_data_path
        )
    


class DataTranformTrain():

    def __init__(self, label, drop_labels_list, perform_cross_validation=True):        
        self.config=DataConfig()
        self.perform_cross_validation = perform_cross_validation
        self.label = label
        self.drop_labels_list = drop_labels_list

    def preprocessor_pipeline(self, df):

        numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']
        categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']

        games_terms = ['_games_overall_', '_games_home_', '_games_away_']
        games_features = [feature for feature in df.columns if any(term in feature for term in games_terms) & (df[feature].dtype != 'O')]

        tbl_terms = ['tbl_']
        tbl_features = [feature for feature in df.columns if any(term in feature for term in tbl_terms) & (df[feature].dtype != 'O')]

        player_terms = ['player_']
        player_features = [feature for feature in df.columns if any(term in feature for term in player_terms) & (df[feature].dtype != 'O')]

        numerical_columns = [x for x in numerical_columns if x not in games_features + tbl_features + player_features]


        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )
        pca_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=5))
            ]
        )
        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        preprocessor=ColumnTransformer(
            [
            ("num_pipeline", num_pipeline, numerical_columns),
            ("games_pca", pca_pipeline, games_features),
            ("tbl_pca", pca_pipeline, tbl_features),
            ("player_pca", pca_pipeline, player_features),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
        )

        return preprocessor


    def algorithms_and_grid(self):

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

        return models, params


    def grid_search(self):
        '''
        todo:
        - add optional print 
        - extract feature importance to own functions
        '''
        models, params = self.algorithms_and_grid()

        train_df=pd.read_csv(self.config.train_data_path)
        test_df=pd.read_csv(self.config.test_data_path)
        val_df=pd.read_csv(self.config.val_data_path)

        X = pd.concat([train_df, test_df]).drop(columns=self.drop_labels_list, axis=1).reset_index(drop=True)
        y = pd.concat([train_df, test_df])[self.label].reset_index(drop=True)

        indices_train = np.arange(train_df.shape[0])
        indices_test = np.arange(test_df.shape[0], train_df.shape[0]+test_df.shape[0])
        cv = [(indices_train, indices_test)]

        if self.perform_cross_validation:
            preprocessor = self.preprocessor_pipeline(df = X)
            indices_train = indices_train.tolist() + indices_test.tolist()
        else:
            preprocessor = self.preprocessor_pipeline(df = train_df.drop(columns=self.drop_labels_list, axis=1))
            indices_train = indices_train.tolist()

        model_list = []
        AUC_ROC_list = []
        algo_best_param = {}
        algo_best_model = []

        mlflow.set_experiment(self.label)
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]

            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model)
            ])

            if self.perform_cross_validation:
                grid = GridSearchCV(estimator = pipeline, param_grid = param, cv = 3, n_jobs = -1, scoring = 'roc_auc', error_score="raise", return_train_score=True)
            else:
                grid = GridSearchCV(estimator = pipeline, param_grid = param, cv = cv, scoring = 'roc_auc', error_score="raise", return_train_score=True)

            grid.fit(X, y)

            cv_results = pd.DataFrame(grid.cv_results_).sort_values(by=["rank_test_score"],ascending=True).reset_index(drop=True)
            cv_results['div'] = cv_results.mean_train_score / cv_results.mean_test_score
            cv_results['ok'] = np.where((cv_results['mean_train_score'] <= 0.95) & (cv_results['div'] <= 1.3), 1, 0)
            cv_results.to_excel('artifacts/ml_results/{0}/{1} - Grid.xlsx'.format(self.label, list(models.keys())[i]), index=False)

            if np.sum(cv_results['ok']) > 0:
                bp = cv_results.loc[cv_results['ok'] == 1]['params'].iloc[0]
            else:
                bp = grid.best_params_

            nbp = {}
            for k, v in bp.items():
                nbp[k[k.index('__')+2:]] = v

            final_pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model.set_params(**nbp))
            ])
            final_pipeline.fit(X.iloc[indices_train], y.iloc[indices_train])
            
            # Evaluate Train and Validation dataset
            metric = evaluate_model_kpi(model=final_pipeline, 
                                        X_train=X.iloc[indices_train], 
                                        y_train=y.iloc[indices_train], 
                                        X_val=val_df, 
                                        y_val=val_df[self.label], 
                                        threshold=0.5,
                                        model_name=list(models.keys())[i]
                                        )
            metric_long = metric.melt(id_vars=['Algorithm'], var_name='Metric', value_name='Metric value')
            metric_long.to_excel('artifacts/ml_results/{0}/{1} - Metrics.xlsx'.format(self.label, list(models.keys())[i]), index=False)

            model_list.append(list(models.keys())[i])
            AUC_ROC_list.append(metric['AUC-ROC Val'][0])

            algo_best_param[list(models.keys())[i]] = nbp

            algo_best_model.append((list(models.keys())[i], final_pipeline))

            # Feature importance

            # Extract feature names
            feature_names = []

            # Get the feature names from the numerical pipeline
            num_feature_names = final_pipeline.named_steps['preprocessing'].transformers_[0][1].get_feature_names_out()
            feature_names.extend(num_feature_names)

            # Get the feature names from the games PCA pipeline
            games_pca_feature_names = ['games_pca_component_' + str(i) for i in range(5)]
            feature_names.extend(games_pca_feature_names)

            # Get the feature names from the tbl PCA pipeline
            tbl_pca_feature_names = ['tbl_pca_component_' + str(i) for i in range(5)]
            feature_names.extend(tbl_pca_feature_names)

            # Get the feature names from the player PCA pipeline
            player_pca_feature_names = ['player_pca_component_' + str(i) for i in range(5)]
            feature_names.extend(player_pca_feature_names)

            # Get the feature names from the categorical pipeline (2023-07-15 - currently I have no categorical features)
            try:
                cat_feature_names = final_pipeline.named_steps['preprocessing'].transformers_[4][1].named_steps['one_hot_encoder'].get_feature_names_out()
                feature_names.extend(cat_feature_names)
            except:
                pass

            if list(models.keys())[i] == "Logistic Regression":
                coefficients = final_pipeline.named_steps['model'].coef_[0]
                fi = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficients': coefficients,
                    'Abs coefficients': np.abs(coefficients)
                })
                fi = fi.sort_values('Abs coefficients', ascending=False).reset_index(drop=True)
            else:
                feature_importance = final_pipeline.named_steps['model'].feature_importances_
                fi = pd.DataFrame({
                    'Feature': feature_names,
                    'Feature importance': feature_importance,
                })
                fi = fi.sort_values('Feature importance', ascending=False).reset_index(drop=True)
            fi.to_excel('artifacts/ml_results/{0}/{1} - Feature importance.xlsx'.format(self.label, list(models.keys())[i]), index=False)

            
            timestamp = " " + str(pd.to_datetime('today'))
            with mlflow.start_run(run_name=list(models.keys())[i] + timestamp):
                mlflow.set_tag('label', self.label)
                mlflow.set_tag('algo', list(models.keys())[i])

                # Log model parameters, metrics, and artifacts using MLflow
                mlflow.log_params(final_pipeline.named_steps['model'].get_params())
                mlflow.log_metric('auc train', metric['AUC-ROC Train'][0])
                mlflow.log_metric('auc val', metric['AUC-ROC Val'][0])
                mlflow.sklearn.log_model(final_pipeline, 'model')

            mlflow.end_run()

        algo_best_model_metric = pd.DataFrame(list(zip(model_list, AUC_ROC_list)), columns=['Model Name', 'AUC_ROC']).sort_values(by=["AUC_ROC"],ascending=False).reset_index(drop=True)
        algo_best_model_metric.to_excel('artifacts/ml_results/{0}/algo_performance.xlsx'.format(self.label), index=False)

        return (
            algo_best_model_metric, 
            algo_best_param, 
            algo_best_model
        )
    
if __name__=='__main__':
    obj=DataIngest()
    train_data,test_data,val_data,score_data=obj.create_train_and_test()

    TransformTrain = DataTranformTrain(label = 'label_1', drop_labels_list = ['label_1', 'label_X', 'label_2'], perform_cross_validation=True)
    algo_best_model_metric, algo_best_param, algo_best_model = TransformTrain.grid_search()

    print(algo_best_model_metric)