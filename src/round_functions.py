import pandas as pd
import requests

def get_round_info(round=None, season=None):
    if season is not None:
        events = pd.read_csv(f"https://raw.githubusercontent.com/wiscostret/histfpldata/master/getroundinfo{season}.csv", encoding="UTF-8")
    else:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        events = pd.DataFrame(response.json()["events"])

    if round is None:
        return events
    else:
        return events[events['id'].isin(round)]
