import requests
import pandas as pd

def get_player_id(name=None):
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    elements = pd.DataFrame(data["elements"])
    elements["playername"] = elements["first_name"] + " " + elements["second_name"]
    if name is None:
        return elements[["playername", "id"]]
    else:
        return elements[elements["playername"].isin(name)][["playername", "id"]]

def get_player_name(playerid=None):
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    elements = pd.DataFrame(data["elements"])
    elements["playername"] = elements["first_name"] + " " + elements["second_name"]
    if playerid is None:
        return elements[["playername", "id"]]
    else:
        return elements[elements["id"].isin(playerid)][["playername", "id"]]

def get_player_info(name=None, season=None):
    if season is None:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        data = response.json()
        elements = pd.DataFrame(data["elements"])
        elements["playername"] = elements["first_name"] + " " + elements["second_name"]
    else:
        try:
            url = f"https://raw.githubusercontent.com/wiscostret/histfpldata/master/getplayerinfo{season}.csv"
            elements = pd.read_csv(url, encoding="UTF-8")
        except:
            pass
    if name is None:
        return elements
    else:
        return elements[elements["playername"].isin(name)]

def get_player_hist(playerid=None):
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    elements = pd.DataFrame(data["elements"])
    elements["playername"] = elements["first_name"] + " " + elements["second_name"]
    histinfo = pd.DataFrame()  # Initialize an empty DataFrame
    if playerid is None:
        for i in range(len(elements)):
            url = f"https://fantasy.premierleague.com/api/element-summary/{elements['id'][i]}/"
            fplboot = requests.get(url).json()["history_past"]
            df = pd.DataFrame(fplboot)
            df["playername"] = elements["playername"][i]
            histinfo = pd.concat([histinfo, df])  # Concatenate the new data with histinfo
    else:
        for pid in playerid:
            url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
            fplboot = requests.get(url).json()["history_past"]
            df = pd.DataFrame(fplboot)
            df["playername"] = elements[elements["id"] == pid]["playername"].values[0]
            histinfo = pd.concat([histinfo, df])  # Concatenate the new data with histinfo
            
    return histinfo


def get_player_details(playerid=None, name=None, season=None):
    if playerid is not None and name is not None:
        raise ValueError("Please only supply playerid OR name, not both.")

    if season is not None:
        try:
            detinfo = pd.read_csv(f"https://raw.githubusercontent.com/wiscostret/histfpldata/master/getplayerdetails{season}.csv", encoding="UTF-8")
        except:
            pass
        
        if playerid is None:
            if name is None:
                return detinfo
            else:
                return detinfo[detinfo['playername'].isin(name)]
        else:
            return detinfo[detinfo['playerid'].isin(playerid)]
    else:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        elements = response.json()["elements"]
        elements_df = pd.DataFrame(elements)
        elements_df['playername'] = elements_df['first_name'] + " " + elements_df['second_name']
        
        if playerid is None:
            if name is None:
                detinfo = pd.DataFrame()
                for i in range(len(elements_df)):
                    response = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{elements_df['id'][i]}/")
                    fplboot = response.json()["history"]
                    detinfo = pd.concat([detinfo, pd.DataFrame(fplboot).assign(playername=elements_df['playername'][i])])
                return detinfo.reset_index(drop=True)
            else:
                detinfo = pd.DataFrame()
                selection = elements_df[elements_df['playername'].isin(name)]['id']
                for playerid in selection:
                    response = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{playerid}/")
                    fplboot = response.json()["history"]
                    detinfo = pd.concat([detinfo, pd.DataFrame(fplboot).assign(playername=elements_df[elements_df['id']==playerid]['playername'].values[0])])
                return detinfo.reset_index(drop=True)
        else:
            detinfo = pd.DataFrame()
            for playerid in playerid:
                response = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{playerid}/")
                fplboot = response.json()["history"]
                detinfo = pd.concat([detinfo, pd.DataFrame(fplboot).assign(playername=elements_df[elements_df['id']==playerid]['playername'].values[0])])
            return detinfo.reset_index(drop=True)
