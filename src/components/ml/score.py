import pandas as pd

def predict_result(model_1, model_X, model_2, predict_data='score'):

    try:
        score = pd.read_csv('artifacts/{0}.csv'.format(predict_data))

        proba_1 = model_1.predict_proba(score)[:, 1]
        proba_X = model_X.predict_proba(score)[:, 1]
        proba_2 = model_2.predict_proba(score)[:, 1]

        score['proba_1_fix'] = proba_1 / (proba_1 + proba_X + proba_2)
        score['proba_X_fix'] = proba_X / (proba_1 + proba_X + proba_2)
        score['proba_2_fix'] = proba_2 / (proba_1 + proba_X + proba_2)

        score[["kickoff_date", "home", "away", "proba_1_fix", "proba_X_fix", "proba_2_fix"]]
    except:
        score = pd.DataFrame()
    
    return score