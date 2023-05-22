import pandas as pd
import requests

def get_game_list(season=None):
    if season is not None:
        url = f"https://raw.githubusercontent.com/wiscostret/histfpldata/master/getgamelist{season}.csv"
        gamelist = pd.read_csv(url, encoding="UTF-8")
        return gamelist
    else:
        fdr_data = get_fdr()
        shorts = fdr_data['short_name']
        
        url = "https://fantasy.premierleague.com/api/fixtures/"
        response = requests.get(url)
        fixtures = response.json()
        
        gamelist = pd.DataFrame({
            "GW": [fixture["event"] for fixture in fixtures],
            "id": [fixture["id"] for fixture in fixtures],
            "home": [fixture["team_h"] for fixture in fixtures],
            "team_h": [fixture["team_h"] for fixture in fixtures],
            "away": [fixture["team_a"] for fixture in fixtures],
            "team_a": [fixture["team_a"] for fixture in fixtures],
            "finished": [fixture["finished"] for fixture in fixtures],
            "kickoff": [fixture["kickoff_time"] for fixture in fixtures],
            "team_h_score": [fixture["team_h_score"] for fixture in fixtures],
            "team_a_score": [fixture["team_a_score"] for fixture in fixtures]
        })
        
        gamelist["home"] = gamelist["home"].map(shorts)
        gamelist["away"] = gamelist["away"].map(shorts)
        
        return gamelist


def get_game_stats(gameid):
    if gameid is None:
        raise ValueError("You'll need to input at least one game ID, mate.")
    if len(gameid) != 1:
        raise ValueError("One at a time, please")

    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    fixtures = response.json()

    game_stats = [fixture["stats"] for fixture in fixtures if fixture["id"] in gameid]
    
    if game_stats:
        return game_stats[0]
    else:
        return None