import requests
import getpass
import pandas as pd

def get_league(leagueid, leaguetype="classic"):
    if leagueid is None:
        raise ValueError("You'll need to input a league ID, mate.")
    if len(leagueid) != 1:
        raise ValueError("One league at a time, please.")
        
    email = input("Please enter your FPL login email: ")
    password = getpass.getpass("Please enter your FPL password: ")
    
    auth_data = {
        "login": email,
        "password": password,
        "redirect_uri": "https://fantasy.premierleague.com/a/login",
        "app": "plfpl-web"
    }
    
    response = requests.post("https://users.premierleague.com/accounts/login/", data=auth_data)
    
    if response.url != "https://fantasy.premierleague.com/a/login?state=success":
        raise ValueError("The authentication didn't work. You've most likely entered an incorrect FPL email and/or password.")
        
    url = f"https://fantasy.premierleague.com/api/leagues-{leaguetype}/{leagueid}/standings/"
    headers = {
        "Authorization": f"Bearer {response.cookies['pl_profile']}"
    }
    league = requests.get(url, headers=headers).json()
    
    return league


def get_league_entries(leagueid, leaguetype="classic", pages=1):
    if leagueid is None:
        raise ValueError("You'll need to input a league ID, mate.")
    if len(leagueid) != 1:
        raise ValueError("One league at a time, please.")
    if pages % 1 != 0:
        raise ValueError("The number of pages needs to be a whole number.")
        
    email = input("Please enter your FPL login email: ")
    password = getpass.getpass("Please enter your FPL password: ")
    
    auth_data = {
        "login": email,
        "password": password,
        "redirect_uri": "https://fantasy.premierleague.com/a/login",
        "app": "plfpl-web"
    }
    
    response = requests.post("https://users.premierleague.com/accounts/login/", data=auth_data)
    
    if response.url != "https://fantasy.premierleague.com/a/login?state=success":
        raise ValueError("The authentication didn't work. You've most likely entered an incorrect FPL email and/or password.")
    
    entries = pd.DataFrame()
    
    for i in range(1, pages+1):
        url = f"https://fantasy.premierleague.com/api/leagues-{leaguetype}/{leagueid}/standings/?page_standings={i}"
        headers = {
            "Authorization": f"Bearer {response.cookies['pl_profile']}"
        }
        standings = requests.get(url, headers=headers).json()
        
        entries = entries.append(pd.DataFrame(standings["standings"]["results"]))
    
    return entries
