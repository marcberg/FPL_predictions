import os
import sys

from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
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
        df_features = df.drop(["season_start_year","GW","id","team_h","team_a","train_score","home","away","kickoff_year","kickoff_month"], axis=1)
        #os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        train_set, test_set=train_test_split(df_features.loc[df.train_score == "train"],test_size=0.4,random_state=42)
        test_set, val_set=train_test_split(test_set,test_size=0.5,random_state=42)

        train_set.to_csv(self.config.train_data_path,index=False,header=True)
        test_set.to_csv(self.config.test_data_path,index=False,header=True)
        val_set.to_csv(self.config.val_data_path,index=False,header=True)
        df.loc[df.train_score == "score"].to_csv(self.config.score_data_path,index=False,header=True)

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

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )

        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
        )

        return preprocessor


    def algorithms_and_grid(self):

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "XGBoost": xgb.XGBClassifier(),
        }

        params = {
            "Decision Tree": {
                #'model__criterion':['log_loss', 'entropy', 'gini'],
                'model__max_depth': [3, 5, 8, 12],
                # 'model__splitter':['best','random'],
                # 'model__max_features':['sqrt','log2'],
            },
            "Random Forest":{
                'model__bootstrap': [True],
                'model__max_depth': [3, 5, 8, 12],
                'model__max_features': [10, 20, 50],
                #'model__min_samples_leaf': [3, 4, 5, 10, 20],
                'model__n_estimators': [50, 100, 500],
            },
            "Gradient Boosting":{
                "model__loss":["log_loss"],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                #"model__min_samples_split": np.linspace(0.1, 0.5, 12),
                #"model__min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "model__min_samples_leaf": [5, 10, 20, 50],
                'model__max_depth': [3, 5, 8, 12],
                #"model__max_features":["log2", "sqrt"],
                #"model__criterion": ["friedman_mse",  "mae"],
                #"model__subsample":[0.5, 0.618, 0.8, 1.0],
                'model__n_estimators': [50, 100, 500],
            },
            "Logistic Regression":{
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'model__penalty': ['l2'],
                'model__max_iter': [10000],
                #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            },
            "XGBoost":{
                'model__max_depth': [3, 5, 8, 12],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                "model__gamma":[0.5, 1, 2],
                'model__n_estimators': [50, 100, 500],
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
            cv_results.to_excel('artifacts/ml_results/{0}/{1} - Grid.xlsx'.format(self.label, list(models.keys())[i]), index=False)

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
            feature_names = final_pipeline.named_steps['preprocessing'].get_feature_names_out()
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