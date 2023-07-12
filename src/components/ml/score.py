import pandas as pd

def extract_best_model(models, models_metric):
    best_model = next(model for name, model in models if name == models_metric['Model Name'][1])
    return best_model

def predict_result(model_1, model_X, model_2):
    score = pd.read_csv('artifacts/score.csv')

    proba_1 = model_1.predict_proba(score)[:, 1]
    proba_X = model_X.predict_proba(score)[:, 1]
    proba_2 = model_2.predict_proba(score)[:, 1]

    score = score[['home', 'away']]
    score['proba_1_fix'] = proba_1 / (proba_1 + proba_X + proba_2)
    score['proba_X_fix'] = proba_X / (proba_1 + proba_X + proba_2)
    score['proba_2_fix'] = proba_2 / (proba_1 + proba_X + proba_2)
    return score[["kickoff_date", "home", "away", "proba_1_fix", "proba_X_fix", "proba_2_fix"]]