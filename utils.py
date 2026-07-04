"""
This file contains utilities to
scrape play-by-play information from
TGL glof matches
"""
import json
import re

import pandas as pd
import numpy as np

def read_json_obj(file_path: str):
    """
    Reads a JSON file and returns the parsed data.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON data.
    """

    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def process_shots(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the shot description to get
    the starting and finishing distance of the shot,
    and the location the shot finishes in.
    """

    def _parse_distance(distance_text: str):
        if not isinstance(distance_text, str) or not distance_text.strip():
            return np.nan

        distance_text = distance_text.strip()
        yd_match = re.match(r"^(?P<yards>\d+)\s*yds?$", distance_text)
        if yd_match:
            return float(yd_match.group("yards"))

        ft_match = re.match(
            r"^(?P<feet>\d+)\s*ft(?:\s*(?P<inches>\d+)\s*in)?$",
            distance_text,
        )
        if ft_match:
            feet = float(ft_match.group("feet"))
            inches = float(ft_match.group("inches") or 0)
            return (feet + inches / 12.0) / 3.0

        in_match = re.match(r"^(?P<inches>\d+)\s*in$", distance_text)
        if in_match:
            inches = float(in_match.group("inches"))
            return inches / 36.0

        return np.nan

    start_distance = []
    end_distance = []
    location = []

    for _, shot in shot_df.iterrows():
        text = str(shot.get("pbpText", ""))
        if "assessed" in text.lower():
            start_distance.append(np.nan)
            end_distance.append(np.nan)
            location.append(np.nan)
            continue

        start_value = np.nan
        end_value = np.nan
        end_location = np.nan

        distance_pattern = r"\d+\s*(?:yds?|ft(?:\s*\d+\s*in)?|in)"

        # standard shot: hits <club> <distance> to <location>, <distance> left to hole
        start_match = re.search(
            rf"\b(?:hits|hit)\b\s+.*?({distance_pattern})\b",
            text,
            re.IGNORECASE,
        )
        if start_match:
            start_value = _parse_distance(start_match.group(1))

        end_match = re.search(
            rf"to\s+([^,]+?),\s*({distance_pattern})\s*left to hole",
            text,
            re.IGNORECASE,
        )
        if end_match:
            end_location = end_match.group(1).strip().title()
            end_value = _parse_distance(end_match.group(2))

        # putts: handle made putts and missed putts
        putt_made_match = re.search(
            rf"makes? putt from\s*({distance_pattern})",
            text,
            re.IGNORECASE,
        )
        putt_miss_match = re.search(
            rf"putts? from\s*({distance_pattern})\s*,\s*({distance_pattern})\s*left to hole",
            text,
            re.IGNORECASE,
        )

        if putt_made_match:
            start_value = _parse_distance(putt_made_match.group(1))
            end_value = 0.0
            end_location = "Hole"
        elif putt_miss_match:
            start_value = _parse_distance(putt_miss_match.group(1))
            end_value = _parse_distance(putt_miss_match.group(2))
            end_location = "Green"

        # if we parsed a hit but not a landing location, try to capture a terminal terrain word
        if pd.isna(end_location) and re.search(r"to\s+(fairway|bunker|rough|green|fringe|tee)\b", text, re.IGNORECASE):
            terrain_match = re.search(r"to\s+(fairway|bunker|rough|green|fringe|tee)\b", text, re.IGNORECASE)
            if terrain_match:
                end_location = terrain_match.group(1).title()

        start_distance.append(start_value)
        end_distance.append(end_value)
        location.append(end_location)

    shot_df = shot_df.copy()
    shot_df["shot_distance"] = start_distance
    shot_df["end_distance"] = end_distance
    shot_df["shot_location"] = location

    return shot_df


def parse_json_data(json_obj: dict):
    """
    This function parses the play-by-play
    data from TGL matches 
    
    Args
        - json_obj (dict): Dictionary of
            play-by-play data for a TGL match

    Returns:
        - pbp_df (pd.DataFrame): DataFrame of
            play-by-play data for TGL matches
    """

    session_list = json_obj["data"]["playByPlayList"]["sessions"]
    half_list = json_obj["data"]['matchDetailsGeoDetect']["sessions"]
    match_id = json_obj["data"]['matchDetailsGeoDetect']["matchId"]
    season_year = json_obj["data"]['matchDetailsGeoDetect']["seasonYear"]
    start_date = json_obj["data"]['matchDetailsGeoDetect']["startDate"]
    overtime = json_obj["data"]['matchDetailsGeoDetect']["overtime"]
    teams = json_obj["data"]['matchDetailsGeoDetect']["teams"]
    team_df = pd.DataFrame()
    players_df = pd.DataFrame()
    for team in teams:
        team_info_df = pd.DataFrame(
            {
                "match_id": [match_id],
                "season_year": [season_year],
                "designation": [team["designation"]],
                "hammers_used": [team["hammersUsed"]],
                "match_probability": [team["matchProbability"]],
                "match_probability_tie": [team["matchProbabilityTie"]],
                "team_id": [team["teamId"]],
                "team_code": [team["teamCode"]],
                "team_name": [team["teamName"]]
            }
        )
        team_df = pd.concat([team_df, team_info_df])
        players = team["players"]
        for player in players:
            player_df = pd.DataFrame(
                {
                    "team_id": [team["teamId"]],
                    "match_id": [match_id],
                    "season_year": [season_year],
                    "player_id": [player["playerId"]],
                    "first_name": [player["firstName"]],
                    "last_name": [player["lastName"]],
                    "is_captain": [player["isCaptain"]]
                }
            )
            players_df = pd.concat([players_df, player_df])


    sessions_df = pd.DataFrame()
    holes_df = pd.DataFrame()
    holes_info_df = pd.DataFrame()
    shots_df = pd.DataFrame()
    for session in session_list:
        session_id = session["sessionId"]
        sequence = session["sequence"]
        session_score = session["sessionScore"]

        session_df = pd.DataFrame(
            {
                "match_id": [match_id],
                "season_year": [season_year],
                "session_id": [session_id],
                "sequence": [sequence],
                "session_score": [session_score]
            }
        )

        session_df["away_score"] = [int(txt.split(" - ")[0]) for txt in session_df["session_score"]]
        session_df["home_score"] = [int(txt.split(" - ")[1]) for txt in session_df["session_score"]]
        sessions_df = pd.concat([sessions_df, session_df])

        holes = session["playByPlay"]
        for hole in holes:
            hole_number = hole["holeNumber"]
            hole_score = hole["holeScore"]
            winning_team_id = hole["holeWinningTeamId"]
            losing_team_id = hole["holeLosingTeamId"]
            shot_df = pd.DataFrame(hole["timeline"]).sort_values("shot", ascending=True)
            shot_df["hole_number"] = hole_number
            shot_df["match_id"] = match_id
            shot_df["season_year"] = season_year
            shot_df = process_shots(shot_df)

            hole_df = pd.DataFrame(
                {
                    "match_id": [match_id],
                    "season_year": [season_year],
                    "hole_number": [hole_number],
                    "hole_score": [hole_score],
                    "session_id": [session_id],
                    "sequence": [sequence],
                    "winning_team_id": [winning_team_id],
                    "losing_team_id": [losing_team_id]
                }
            )
            shots_df = pd.concat([shots_df, shot_df])
            holes_df = pd.concat([holes_df, hole_df])
        
        for half in half_list:
            holes = half['holes']
            for hole in holes:
                hole_info_df = pd.DataFrame(
                    {
                        "match_id": [match_id],
                        "season_year": [season_year],
                        "hole_config_id": [hole["holeConfigId"]],
                        "hole_id": [hole["holeId"]],
                        "hole_name": [hole["holeName"]],
                        "hole_number": [hole["holeNumber"]],
                        "hole_par": [hole["holePar"]],
                        "hole_value": [hole["holeValue"]],
                        "yards": [hole["yards"]]
                    }
                )
                holes_info_df = pd.concat([holes_info_df, hole_info_df])

    return sessions_df, holes_df, shots_df, holes_info_df, team_df, players_df