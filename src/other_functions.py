import requests
import pandas as pd

def get_fdr():
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    teams = data["teams"]
    return teams