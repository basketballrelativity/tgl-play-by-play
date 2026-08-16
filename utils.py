"""
This file contains utilities to
scrape play-by-play information from
TGL glof matches
"""
import json
import re
import pickle

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


def convert_distance_to_yards(distance_value):
    """Convert a standardized golf distance string into yards.

    Supported examples include:
      - "120 yds" -> 120.0
      - "79 ft 6 in." -> 26.5
      - "5 ft 2 in" -> 1.722222...
      - "18 in." -> 0.5
      - "--" -> 0.0
    """
    if pd.isna(distance_value):
        return np.nan

    value = str(distance_value).strip()
    if not value:
        return np.nan

    value = re.sub(r"\s+", " ", value)
    value = value.rstrip(".")

    if value in {"--", "-", "—"}:
        return 0.0

    yd_match = re.fullmatch(
        r"(?P<yards>\d+(?:\.\d+)?)\s*(?:yd|yds)",
        value,
        re.IGNORECASE,
    )
    if yd_match:
        return float(yd_match.group("yards"))

    ft_in_match = re.fullmatch(
        r"(?P<feet>\d+(?:\.\d+)?)\s*ft\s+(?P<inches>\d+(?:\.\d+)?)\s*in",
        value,
        re.IGNORECASE,
    )
    if ft_in_match:
        feet = float(ft_in_match.group("feet"))
        inches = float(ft_in_match.group("inches"))
        return (feet + inches / 12.0) / 3.0

    ft_match = re.fullmatch(r"(?P<feet>\d+(?:\.\d+)?)\s*ft", value, re.IGNORECASE)
    if ft_match:
        return float(ft_match.group("feet")) / 3.0

    in_match = re.fullmatch(r"(?P<inches>\d+(?:\.\d+)?)\s*in", value, re.IGNORECASE)
    if in_match:
        return float(in_match.group("inches")) / 36.0

    numeric_match = re.fullmatch(r"\d+(?:\.\d+)?", value)
    if numeric_match:
        return float(value)

    return np.nan


def convert_distance_column(distance_series: pd.Series) -> pd.Series:
    """Apply the yard conversion to an entire distance column."""
    return distance_series.apply(convert_distance_to_yards)


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

        # putts: handle made putts, missed putts, holing out, and chip shots
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
        hole_out_match = re.search(
            rf"holes? out from\s*({distance_pattern})",
            text,
            re.IGNORECASE,
        )
        chip_match = re.search(
            rf"chips? from\s*({distance_pattern})\s*,\s*({distance_pattern})\s*left to hole",
            text,
            re.IGNORECASE,
        )

        if hole_out_match:
            start_value = _parse_distance(hole_out_match.group(1))
            end_value = 0.0
            end_location = "Hole"
        elif putt_made_match:
            start_value = _parse_distance(putt_made_match.group(1))
            end_value = 0.0
            end_location = "Hole"
        elif putt_miss_match:
            start_value = _parse_distance(putt_miss_match.group(1))
            end_value = _parse_distance(putt_miss_match.group(2))
            end_location = "Green"
        elif chip_match:
            start_value = _parse_distance(chip_match.group(1))
            end_value = _parse_distance(chip_match.group(2))
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

        session_df["away_score"] = [int(txt.split(" - ")[0]) if pd.notnull(txt) and " - " in txt else None for txt in session_df["session_score"]]
        session_df["home_score"] = [int(txt.split(" - ")[1]) if pd.notnull(txt) and " - " in txt else None for txt in session_df["session_score"]]
        sessions_df = pd.concat([sessions_df, session_df])

        holes = session["playByPlay"]
        for hole in holes:
            hole_number = hole["holeNumber"]
            hole_score = hole["holeScore"]
            winning_team_id = hole["holeWinningTeamId"]
            losing_team_id = hole["holeLosingTeamId"]
            shot_df = pd.DataFrame(hole["timeline"])
            if len(shot_df) > 0:
                shot_df = shot_df.sort_values("shot", ascending=True)
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


def get_drive_ex_strokes(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function derives the expected strokes from the shot-level data
    for drives to serve a downstream hole win probability model.

    Args:
        shot_df (pd.DataFrame): A DataFrame containing the shot-level data.
    
    Returns:
        pd.DataFrame: A DataFrame with expected strokes
            off the tee only
    """

    # Unpack drive models
    # Just copying these from the notebook
    drive_models = {
        3: {"intercept": 2.5686469023363045, "slope": 0.00254143},
        4: {"intercept": 3.187367319289582, "slope": 0.00192602},
        5: {"intercept": 2.7921565590168576, "slope": 0.00326362} 
    }

    # Filter for drive shots (assuming drive shots are the first shot of each hole)
    shot_df["drive_ex_strokes"] = [
        np.nan if pd.isnull(shot_number)
        else np.nan if shot_number != 1
        else drive_models[par]["intercept"] + drive_models[par]["slope"] * distance
        for shot_number, par, distance in zip(
            shot_df["shot_number"],
            shot_df["hole_par"],
            shot_df["yards"]
        )
    ]

    return shot_df


def get_putt_ex_strokes(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function derives the expected strokes from the shot-level data
    for putts to serve a downstream hole win probability model.

    Args:
        shot_df (pd.DataFrame): A DataFrame containing the shot-level data.
    
    Returns:
        pd.DataFrame: A DataFrame with expected strokes
            for putts only
    """

    # Unpack putt models
    with open('putt_model.pkl', 'rb') as file:
        putt_models = pickle.load(file)

    def calc_ex_strokes(putt_models, distance):
        """
        Calculate expected strokes for a putt based on the distance.
        """

        distance_ft = distance * 3.0  # Convert yards to feet

        # Unpack model parameters
        intercept = putt_models["coef"]["Intercept"]
        slope = putt_models["coef"]["distance_float"]
        spline_coef = np.array(putt_models["coef"].iloc[2:])

        # Transform distance using spline basis functions
        distance_transformed = putt_models["splines"].transform(np.array(distance_ft).reshape(-1, 1))

        # Calculate expected strokes
        ex_strokes = intercept + (slope * distance_ft) + np.dot(spline_coef, distance_transformed.T)

        return np.exp(ex_strokes)[0] + 1  # Add 1 to account for the current putt
        

    # Filter for putt shots (assuming putts are the last shot of each hole)
    shot_df["putt_ex_strokes"] = [
        np.nan if shot_location != "Green"
        else calc_ex_strokes(putt_models, distance)
        for shot_location, distance in zip(
            shot_df["shot_location"],
            shot_df["end_distance"]
            )
    ]

    return shot_df


def get_approach_ex_strokes(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function derives the expected strokes from the shot-level data
    for approach shots to serve a downstream hole win probability model.

    Args:
        shot_df (pd.DataFrame): A DataFrame containing the shot-level data.
    
    Returns:
        pd.DataFrame: A DataFrame with expected strokes
            for approach shots only
    """

    # Unpack approach models
    with open('approach_objs.pkl', 'rb') as file:
        approach_models = pickle.load(file)

    approach_df = shot_df[
        (shot_df["shot_location"].isin(["Fairway", "Rough", "Bunker", "Free Drop Area"])) &
        (shot_df["strokeType"] == "SHOT")
    ]
    features = ["distance_norm", "fairway", "rough", "bunker", "native_area", "other"]
    approach_df["distance_norm"] = approach_models["distance_scaler"].transform(np.array(approach_df["end_distance"]).reshape(-1, 1))
    approach_df["fairway"] = [1 if loc in ["Fairway", "Free Drop Area"] else 0 for loc in approach_df["shot_location"]]
    approach_df["rough"] = [1 if loc == "Rough" else 0 for loc in approach_df["shot_location"]]
    approach_df["bunker"] = [1 if loc == "Bunker" else 0 for loc in approach_df["shot_location"]]
    approach_df["native_area"] = [1 if loc == "Native Area" else 0 for loc in approach_df["shot_location"]]
    approach_df["other"] = [1 if loc not in ["Fairway", "Rough", "Bunker", "Native Area"] else 0 for loc in approach_df["shot_location"]]

    approach_df["approach_ex_strokes"] = approach_models["model"].predict(approach_df[features])

    # Join back approach shots
    shot_df = shot_df.merge(approach_df[
        ["match_id", "hole_config_id", "hole_id", "hole_number",
         "sequence", "shot_number", "playerId", "teamId", "approach_ex_strokes"]
    ],
    on=["match_id", "hole_config_id", "hole_id", "hole_number",
        "sequence", "shot_number", "playerId", "teamId"],
    how="left")

    return shot_df


def process_shot_data(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the shot-level data to derive expected strokes
    for drives, putts, and approach shots.

    Args:
        shot_df (pd.DataFrame): A DataFrame containing the shot-level data.
    
    Returns:
        pd.DataFrame: A DataFrame with expected strokes for drives, putts, and approach shots.
    """

    shot_df = get_drive_ex_strokes(shot_df)
    shot_df = get_putt_ex_strokes(shot_df)
    shot_df = get_approach_ex_strokes(shot_df)

    unique_holes = list(set(shot_df["hole_number"]))
    strokes_df = pd.DataFrame()

    for hole in unique_holes:
        hole_shots = shot_df[shot_df["hole_number"] == hole]
        hole_shots = hole_shots.sort_values("shot_number", ascending=True)
        unique_teams = list(set(hole_shots["teamId"]))
        for team in unique_teams:
            team_shots = hole_shots[hole_shots["teamId"] == team]
            ex_strokes = []
            shot_number = []
            for _, shot in team_shots.iterrows():
                shot_number.append(shot["shot_number"])
                if pd.notnull(shot["shot_number"]) and shot["shot_number"] == 1:
                    ex_strokes.append(shot["drive_ex_strokes"])
                    next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else shot["putt_ex_strokes"]
                else:
                    ex_strokes.append(next_stroke)
                    next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else shot["putt_ex_strokes"]
            team_hole_df = pd.DataFrame(
                {
                    "ex_strokes": ex_strokes,
                    "shot_number": shot_number
                }
            )
            team_hole_df["teamId"] = team
            team_hole_df["hole_number"] = hole
            strokes_df = pd.concat([strokes_df, team_hole_df])

    shot_df = shot_df.merge(
        strokes_df,
        on=["teamId", "hole_number", "shot_number"],
        how="left"
    )

    return shot_df