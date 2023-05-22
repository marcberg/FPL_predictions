import requests

def get_entry(entryid):
    if entryid is None:
        raise ValueError("You'll need to input at least one entry ID, mate.")
    if len(entryid) != 1:
        raise ValueError("One at a time, please")
    
    url = f"https://fantasy.premierleague.com/api/entry/{entryid}/"
    response = requests.get(url)
    entry = response.json()
    return entry

def get_entry_hist(entryid):
    if entryid is None:
        raise ValueError("You'll need to input at least one entry ID, mate.")
    
    entryhistory2 = []
    for entry_id in entryid:
        entry_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/"
        entryhistory_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/"
        
        entry_response = requests.get(entry_url)
        entry = entry_response.json()
        
        entryhistory_response = requests.get(entryhistory_url)
        entryhistory = entryhistory_response.json()
        
        if len(entryhistory["past"]) == 0:
            entryhistory["past"] = {"season_name": "", "total_points": "", "rank": ""}
        
        entryhistory2.append({"name": f"{entry['player_first_name']} {entry['player_last_name']}",
                              "season_name": entryhistory["past"]["season_name"],
                              "total_points": entryhistory["past"]["total_points"],
                              "rank": entryhistory["past"]["rank"]})
    
    return entryhistory2

def get_entry_season(entryid):
    if entryid is None:
        raise ValueError("You'll need to input at least one entry ID, mate.")
    
    entryseason2 = []
    for entry_id in entryid:
        entry_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/"
        entryseason_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/"
        
        entry_response = requests.get(entry_url)
        entry = entry_response.json()
        
        entryseason_response = requests.get(entryseason_url)
        entryseason = entryseason_response.json()
        
        entryseason2.append({"name": f"{entry['player_first_name']} {entry['player_last_name']}",
                             "current": entryseason["current"]})
    
    return entryseason2

def get_entry_picks(entryid, gw):
    if entryid is None:
        raise ValueError("You'll need to input an entry ID, mate.")
    if gw is None:
        raise ValueError("You'll need to input a gameweek, mate.")
    
    url = f"https://fantasy.premierleague.com/api/entry/{entryid}/event/{gw}/picks/"
    response = requests.get(url)
    picks = response.json()
    return picks
