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

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

import joblib

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

        print("create_train_and_test - Creating train, test, val and score")
        df=pd.read_csv(self.config.raw_data_path)

        train_set, test_set=train_test_split(df.loc[df.train_score == "train"],test_size=0.4,random_state=42)
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
        teams_next_game = score.merge(pd.DataFrame(next_games), on="id", how="inner")
        teams_next_game.to_csv(self.config.score_data_path,index=False,header=True)

        print("create_train_and_test - DONE! \n")
        return(
            self.config.train_data_path,
            self.config.test_data_path,
            self.config.val_data_path,
            self.config.score_data_path
        )
    
def test_significant(df, feature, target):
    
    if is_string_dtype(df[feature]) or (is_numeric_dtype(df[feature]) and len(df[feature].unique()) <= 5):

        df_temp = pd.DataFrame({"feature": df[feature], "target": df[target]})

        # Perform one-way ANOVA
        model = smf.ols('target ~ feature', data=df_temp).fit()
        anova_table = sm.stats.anova_lm(model)

        result = pd.DataFrame({"feature": [feature], "PR(>F)": [anova_table['PR(>F)'][0]]})
        
    elif is_numeric_dtype(df[feature]):

        bins = np.nanpercentile(df[feature], [0, 20, 40, 60, 80, 100])
        bins = [i for n, i in enumerate(bins) if i not in bins[:n]]
        bins[0] = np.floor(bins[0])
        bins[-1] = np.ceil(bins[-1])
                
        df_temp = pd.DataFrame({"feature": df[feature], "target": df[target]})
        df_temp[feature+'_bins'] = pd.cut(pd.to_numeric(df[feature]), bins, include_lowest=True)

        # Perform one-way ANOVA
        model = smf.ols(f"target ~ {feature+'_bins'}", data=df_temp).fit()
        anova_table = sm.stats.anova_lm(model)

        result = pd.DataFrame({"feature": [feature], "PR(>F)": [anova_table['PR(>F)'][0]]})
    
    return result

class DataTranformTrain():

    def __init__(self, label, perform_cross_validation=True):        
        self.config=DataConfig()
        self.label = label
        self.perform_cross_validation = perform_cross_validation


    def feature_selection_by_test(self, select_p_value=0.05):

        print("- Calculating significant features")
        train_df=pd.read_csv(self.config.train_data_path)

        if self.perform_cross_validation:
            test_df=pd.read_csv(self.config.test_data_path)
            train_df = pd.concat([train_df, test_df]).reset_index(drop=True)

        numerical_columns = [feature for feature in train_df.columns if train_df[feature].dtype != 'O']
        categorical_columns = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']

        features = numerical_columns + categorical_columns

        p_values = pd.DataFrame()
        for p in features:
            p_value = test_significant(train_df, p, self.label)
            p_values = pd.concat([p_values, p_value]).reset_index(drop=True)

        selected_features = p_values.loc[p_values["PR(>F)"] < select_p_value]["feature"].to_list()
        return selected_features
    

    def preprocessor_pipeline(self, df):

        dont_use_feature = ["season_start_year","GW","id","team_h","team_a","train_score","home","away","kickoff_year","kickoff_month","kickoff_date", 'label_1', 'label_X', 'label_2']
        
        categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']
        categorical_columns = [x for x in categorical_columns if x not in dont_use_feature]

        games_terms = ['_games_overall_', '_games_home_', '_games_away_']
        games_features = [feature for feature in df.columns if any(term in feature for term in games_terms) & (df[feature].dtype != 'O')]
        games_features = [x for x in games_features if x not in dont_use_feature]

        tbl_terms = ['tbl_']
        tbl_features = [feature for feature in df.columns if any(term in feature for term in tbl_terms) & (df[feature].dtype != 'O')]
        tbl_features = [x for x in tbl_features if x not in dont_use_feature]

        player_terms = ['player_']
        player_features = [feature for feature in df.columns if any(term in feature for term in player_terms) & (df[feature].dtype != 'O')]
        player_features = [x for x in player_features if x not in dont_use_feature]

        selected_numerical_columns = self.feature_selection_by_test()
        selected_numerical_columns = [x for x in selected_numerical_columns if x not in dont_use_feature]

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
            ("num_pipeline", num_pipeline, selected_numerical_columns),
            ("games_pca", pca_pipeline, games_features),
            ("tbl_pca", pca_pipeline, tbl_features),
            ("player_pca", pca_pipeline, player_features),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
        )

        return preprocessor


    def adjust_train_test(self):
        print(self.label)

        print("- Importing train, test and val")
        train_df=pd.read_csv(self.config.train_data_path)
        test_df=pd.read_csv(self.config.test_data_path)
        val_df=pd.read_csv(self.config.val_data_path)

        X = pd.concat([train_df, test_df]).reset_index(drop=True)
        y = pd.concat([train_df, test_df])[self.label].reset_index(drop=True)

        indices_train = np.arange(train_df.shape[0])
        indices_test = np.arange(test_df.shape[0], train_df.shape[0]+test_df.shape[0])
        cv = [(indices_train, indices_test)]

        if self.perform_cross_validation:
            preprocessor = self.preprocessor_pipeline(df = X)
            indices_train = indices_train.tolist() + indices_test.tolist()
        else:
            preprocessor = self.preprocessor_pipeline(df = train_df)
            indices_train = indices_train.tolist()

        return X, y, indices_train, indices_test, cv, preprocessor, val_df
    

    def select_param_from_grid(self, grid, model_name):

        cv_results = pd.DataFrame(grid.cv_results_).sort_values(by=["rank_test_score"],ascending=True).reset_index(drop=True)
        cv_results['div'] = cv_results.mean_train_score / cv_results.mean_test_score
        cv_results['ok'] = np.where((cv_results['mean_train_score'] <= 0.95) & (cv_results['div'] <= 1.3), 1, 0)
        cv_results.to_excel('artifacts/ml_results/{0}/{1} - Grid.xlsx'.format(self.label, model_name), index=False)

        if np.sum(cv_results['ok']) > 0:
            bp = cv_results.loc[cv_results['ok'] == 1]['params'].iloc[0]
        else:
            bp = grid.best_params_

        nbp = {}
        for k, v in bp.items():
            nbp[k[k.index('__')+2:]] = v

        return nbp


    def calculate_and_save_metrics(self, final_pipeline, X, y, indices_train, val_df, model_name):
        metric = evaluate_model_kpi(model=final_pipeline, 
                                        X_train=X.iloc[indices_train], 
                                        y_train=y.iloc[indices_train], 
                                        X_val=val_df, 
                                        y_val=val_df[self.label], 
                                        threshold=0.5,
                                        model_name=model_name
                                        )
        metric_long = metric.melt(id_vars=['Algorithm'], var_name='Metric', value_name='Metric value')
        metric_long.to_excel('artifacts/ml_results/{0}/{1} - Metrics.xlsx'.format(self.label, model_name), index=False)
        print(metric_long[["Metric", "Metric value"]])
        print("\n")
        
        return metric


    def collect_and_save_feature_importance(self, final_pipeline, model_name):
        
        feature_names = []

        num_feature_names = final_pipeline.named_steps['preprocessing'].transformers_[0][1].get_feature_names_out()
        feature_names.extend(num_feature_names)

        games_pca_feature_names = ['games_pca_component_' + str(i) for i in range(5)]
        feature_names.extend(games_pca_feature_names)

        tbl_pca_feature_names = ['tbl_pca_component_' + str(i) for i in range(5)]
        feature_names.extend(tbl_pca_feature_names)

        player_pca_feature_names = ['player_pca_component_' + str(i) for i in range(5)]
        feature_names.extend(player_pca_feature_names)

        # Get the feature names from the categorical pipeline (2023-07-15 - currently I have no categorical features)
        try:
            cat_feature_names = final_pipeline.named_steps['preprocessing'].transformers_[4][1].named_steps['one_hot_encoder'].get_feature_names_out()
            feature_names.extend(cat_feature_names)
        except:
            pass

        if model_name == "Logistic Regression":
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
        fi.to_excel('artifacts/ml_results/{0}/{1} - Feature importance.xlsx'.format(self.label, model_name), index=False)


    def grid_search(self, models, params, save_to_mlflow=True):

        X, y, indices_train, indices_test, cv, preprocessor, val_df = self.adjust_train_test()

        mlflow.set_experiment(self.label)

        print("- Hyperparameter-tuning and training best model for each algo: \n")
        algo_best_model = []
        all_algo_metrics = pd.DataFrame()
        for i in range(len(list(models))):
            print(list(models.keys())[i])

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
            
            print("- Hyperparameter-tuning")
            grid.fit(X, y)


            print("- Select best hyperparameters")
            nbp = self.select_param_from_grid(grid=grid, model_name=list(models.keys())[i])

            print("- Train best model")
            final_pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model.set_params(**nbp))
            ])
            final_pipeline.fit(X.iloc[indices_train], y.iloc[indices_train])

            # Save the pipeline to a file
            joblib.dump(final_pipeline, 'artifacts/ml_results/{0}/{1}.pkl'.format(self.label, list(models.keys())[i]))

            algo_best_model.append((list(models.keys())[i], final_pipeline))        

            print("- Calculating metrics \n")
            metric = self.calculate_and_save_metrics(final_pipeline, X, y, indices_train, val_df, model_name=list(models.keys())[i])
            all_algo_metrics = pd.concat([all_algo_metrics, metric]).reset_index(drop=True)

            print("- Collect feature importance \n")
            self.collect_and_save_feature_importance(final_pipeline, model_name=list(models.keys())[i])
            
            if save_to_mlflow:
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

        all_algo_metrics.to_excel('artifacts/ml_results/{0}/all_algo_metrics.xlsx'.format(self.label), index=False)
    
    
if __name__=='__main__':
    obj=DataIngest()
    train_data,test_data,val_data,score_data=obj.create_train_and_test()

    TransformTrain = DataTranformTrain(label = 'label_1', perform_cross_validation=True)
    algo_best_model_metric, algo_best_param, algo_best_model = TransformTrain.grid_search()

    print(algo_best_model_metric)