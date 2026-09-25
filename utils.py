"""
This file contains utilities to
scrape play-by-play information from
TGL glof matches
"""
import json
import re
import pickle
from typing import List

import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

from pygam import LogisticGAM, te, s


import sg_data
import sg_utils


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


def process_shots(shot_df: pd.DataFrame):
    """
    This function processes the shot description to get
    the starting and finishing distance of the shot,
    and the location the shot finishes in.

    Args:
        shot_df (pd.DataFrame): DataFrame containing shot-level
            data

    Returns:
        - None if a valid distance value can't be found, otherwise
            distance in units of yards
    """

    def _parse_distance(distance_text: str):
        if not isinstance(distance_text, str) or not distance_text.strip():
            return np.nan

        # Search for yds and return if found
        distance_text = distance_text.strip()
        yd_match = re.match(r"^(?P<yards>\d+)\s*yds?$", distance_text)
        if yd_match:
            return float(yd_match.group("yards"))

        # Search for feet and inches
        ft_match = re.match(
            r"^(?P<feet>\d+)\s*ft(?:\s*(?P<inches>\d+)\s*in)?$",
            distance_text,
        )
        # Return if found
        if ft_match:
            feet = float(ft_match.group("feet"))
            inches = float(ft_match.group("inches") or 0)
            return (feet + inches / 12.0) / 3.0

        # Search for inches only
        in_match = re.match(r"^(?P<inches>\d+)\s*in$", distance_text)
        if in_match:
            inches = float(in_match.group("inches"))
            return inches / 36.0

        return np.nan

    # Initialize lists for storage of start and end distance,
    # along with where the shot ends up (location)
    start_distance = []
    end_distance = []
    location = []

    # Loop through shots
    for _, shot in shot_df.iterrows():
        text = str(shot.get("pbpText", ""))
        
        # Is "assessed" is in the pbp text, this is a penalty stroke
        if "assessed" in text.lower():
            start_distance.append(np.nan)
            end_distance.append(np.nan)
            location.append(np.nan)
            continue

        # Initialize with NaN
        start_value = np.nan
        end_value = np.nan
        end_location = np.nan

        # Core distance pattern
        distance_pattern = r"\d+\s*(?:yds?|ft(?:\s*\d+\s*in)?|in)"

        # standard shot: hits <club> <distance> to <location>, <distance> left to hole
        start_match = re.search(
            rf"\b(?:hits|hit)\b\s+.*?({distance_pattern})\b",
            text,
            re.IGNORECASE,
        )
        # Parses start distance (which is really shot distance)
        if start_match:
            start_value = _parse_distance(start_match.group(1))

        # Insert the core distance pattern after "to"
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

        # Handle the differnt putt or chip outcomes
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

    # Store the distance values and return the DataFrame
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

    # Unpack the data stored in various lists
    session_list = json_obj["data"]["playByPlayList"]["sessions"]
    half_list = json_obj["data"]['matchDetailsGeoDetect']["sessions"]
    match_id = json_obj["data"]['matchDetailsGeoDetect']["matchId"]
    season_year = json_obj["data"]['matchDetailsGeoDetect']["seasonYear"]
    teams = json_obj["data"]['matchDetailsGeoDetect']["teams"]

    # Initialize team and player DataFrames
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

    # Initialize session, hole, and shot DataFrames
    sessions_df = pd.DataFrame()
    holes_df = pd.DataFrame()
    holes_info_df = pd.DataFrame()
    shots_df = pd.DataFrame()

    # Two sessions per match (triples followed by singles)
    for session in session_list:
        session_id = session["sessionId"]
        sequence = session["sequence"]
        session_score = session["sessionScore"]

        # Note that the second session score is cumulative
        # (includes the first session score)
        session_df = pd.DataFrame(
            {
                "match_id": [match_id],
                "season_year": [season_year],
                "session_id": [session_id],
                "sequence": [sequence],
                "session_score": [session_score]
            }
        )

        # Separate out home and away scores
        session_df["away_score"] = [int(txt.split(" - ")[0]) if pd.notnull(txt) and " - " in txt else None for txt in session_df["session_score"]]
        session_df["home_score"] = [int(txt.split(" - ")[1]) if pd.notnull(txt) and " - " in txt else None for txt in session_df["session_score"]]
        sessions_df = pd.concat([sessions_df, session_df])

        # Loop through each hole
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
                # Extract shot distance, along with end distance and
                # location
                shot_df = process_shots(shot_df)

            # Store hole information
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
        
        # Loop through each session
        for half in half_list:
            holes = half['holes']
            for hole in holes:
                # Extract and store hole information
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

    # Return it all, cowboy!
    return sessions_df, holes_df, shots_df, holes_info_df, team_df, players_df


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

    def calc_putts(putt_models, distance, putt_number):
        """
        Calculate expected strokes for a putt based on the distance.
        """
        assert putt_number in [1, 3] 
        suffix = "_p" if putt_number == 1 else "_t"

        distance_ft = distance * 3.0  # Convert yards to feet

        # Unpack model parameters
        intercept = putt_models["coef" + suffix]["Intercept"]
        slope = putt_models["coef" + suffix]["distance_float"]
        spline_coef = np.array(putt_models["coef" + suffix].iloc[2:])

        # Transform distance using spline basis functions
        distance_transformed = putt_models["splines"].transform(np.array(distance_ft).reshape(-1, 1))

        # Calculate expected strokes
        log_odds = intercept + (slope * distance_ft) + np.dot(spline_coef, distance_transformed.T)

        return expit(log_odds)[0]  # Add 1 to account for the current putt
        

    # Filter for putt shots (assuming putts are the last shot of each hole)
    shot_df["one_putt"] = [
        np.nan if shot_location != "Green"
        else calc_putts(putt_models, distance, 1)
        for shot_location, distance in zip(
            shot_df["shot_location"],
            shot_df["end_distance"]
            )
    ]

    shot_df["three_putt"] = [
            np.nan if shot_location != "Green"
            else calc_putts(putt_models, distance, 3)
            for shot_location, distance in zip(
                shot_df["shot_location"],
                shot_df["end_distance"]
                )
        ]

    shot_df["putt_ex_strokes"] = shot_df["one_putt"] + shot_df["three_putt"]*3 + 2*(1 - (shot_df["one_putt"] + shot_df["three_putt"]))

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
            approach_model = pickle.load(file)

    approach_df = shot_df[
        (shot_df["shot_location"].isin(["Fairway", "Rough", "Bunker", "Native Area", "Free Drop Area", "Penalty Area"])) &
        (shot_df["strokeType"] == "SHOT")
    ]
    features = ["distance_norm", "fairway", "rough", "bunker", "native_area", "other"]
    approach_df["distance_norm"] = approach_model["distance_scaler"].transform(np.array(approach_df["end_distance"]).reshape(-1, 1))
    approach_df["fairway"] = [1 if loc in ["Fairway", "Free Drop Area"] else 0 for loc in approach_df["shot_location"]]
    approach_df["rough"] = [1 if loc == "Rough" else 0 for loc in approach_df["shot_location"]]
    approach_df["bunker"] = [1 if loc == "Bunker" else 0 for loc in approach_df["shot_location"]]
    approach_df["native_area"] = [1 if loc == "Native Area" else 0 for loc in approach_df["shot_location"]]
    approach_df["other"] = [1 if loc not in ["Fairway", "Rough", "Bunker", "Native Area", "Free Drop Area"] else 0 for loc in approach_df["shot_location"]]

    approach_df["approach_ex_strokes"] = approach_model["ex_strokes_model"].predict(approach_df[features])

    preds = approach_model["prob_model"].predict(approach_df[["approach_ex_strokes", "fairway", "bunker", "rough"]])
    strokes = np.array(range(1, len(preds.columns)+1))
    for stroke in strokes:
        approach_df[f"{stroke}_app_stroke_prob"] = preds[stroke-1]

    shot_df = shot_df.merge(approach_df[
        ["match_id", "hole_config_id", "hole_id", "hole_number",
         "sequence", "shot_number", "playerId", "teamId", "approach_ex_strokes"] + [col for col in approach_df.columns if col.endswith('_app_stroke_prob')]
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

    shot_df = sg_utils.get_drive_ex_strokes(shot_df)
    shot_df = get_putt_ex_strokes(shot_df)
    shot_df = get_approach_ex_strokes(shot_df)

    unique_holes = list(set(shot_df["hole_number"]))
    strokes_df = pd.DataFrame()

    for hole in unique_holes:
        hole_shots = shot_df[shot_df["hole_number"] == hole]
        hole_shots = hole_shots.sort_values("shot_number", ascending=True)
        unique_teams = list(set(hole_shots["teamId"]))
        for team in unique_teams:
            team_shots = hole_shots[(hole_shots["teamId"] == team)]
            ex_strokes = []
            one_putt_prob = []
            three_putt_prob = []
            shot_number = []
            stroke_prob_rows = []
            for _, shot in team_shots.iterrows():
                shot_number.append(shot["shot_number"])
                if pd.notnull(shot["shot_number"]) and shot["shot_number"] == 1:
                    drive_prob_cols = list(dict.fromkeys(
                        col for col in shot_df.columns if "drive_stroke_prob" in col
                    ))
                    app_prob_cols = list(dict.fromkeys(
                        col for col in shot_df.columns if "app_stroke_prob" in col
                    ))
                    ex_strokes.append(shot["drive_ex_strokes"])
                    one_putt_prob.append(np.nan)
                    three_putt_prob.append(np.nan)
                    next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else 0 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"])=="GIMME" else shot["putt_ex_strokes"]
                    next_one_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else 1 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"]) == "GIMME" else shot["one_putt"]
                    next_three_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else 0 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"]) == "GIMME" else shot["three_putt"]
                    stroke_prob_rows.append({
                        col.replace("drive_", ""): shot[col]
                        for col in drive_prob_cols
                    })
                    next_probs = {
                        col.replace("app_", ""): shot[col]
                        for col in app_prob_cols
                    }
                else:
                    ex_strokes.append(next_stroke)
                    one_putt_prob.append(next_one_putt)
                    three_putt_prob.append(next_three_putt)
                    stroke_prob_rows.append(next_probs)

                    if shot["strokeType"] != "PENALTY":
                        next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else 0 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"]) == "GIMME" else shot["putt_ex_strokes"]
                        next_one_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else 1 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"]) == "GIMME" else shot["one_putt"]
                        next_three_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else 0 if str(shot["shot_location"]) == "Hole" or str(shot["strokeType"]) == "GIMME" else shot["three_putt"]
                        next_probs = {
                            col.replace("app_", ""): shot[col]
                            for col in app_prob_cols
                        }

            prob_df = pd.DataFrame(stroke_prob_rows)

            team_hole_df = pd.DataFrame(
                {
                    "ex_strokes": ex_strokes,
                    "one_putt_prob": one_putt_prob,
                    "three_putt_prob": three_putt_prob,
                    "shot_number": shot_number
                }
            )
            team_hole_df["teamId"] = team
            team_hole_df["hole_number"] = hole
            for col in prob_df.columns:
                team_hole_df[col] = prob_df[col].values
            strokes_df = pd.concat([strokes_df, team_hole_df])

    for col in prob_df.columns:
        if col in shot_df.columns:
            del shot_df[col]
    shot_df = shot_df.merge(
        strokes_df,
        on=["teamId", "hole_number", "shot_number"],
        how="left"
    )

    return shot_df


def get_probability_vectors(shot: pd.Series, team: str):
    """
    This function calculates the probability vectors for the shooting team and the other team
    based on the expected strokes and putt probabilities.

    Args:
        shot (pd.Series): A pandas Series containing the shot-level data for a specific shot.
        team (string): One of "shooting" or "other"

    Returns:
        tuple: A tuple containing two lists:
            - team_probs (list): Probability vector for the provided team.
    """

    if shot[f"{team}_team_ex_strokes"] == 0:
        team_probs = [np.nan] * 10  # Already holed out
    elif pd.notnull(shot[f"{team}_team_one_putt_prob"]):
        two_putt_prob = 1 - shot[f"{team}_team_one_putt_prob"] - shot[f"{team}_team_three_putt_prob"]
        team_probs = [
            shot[f"{team}_team_one_putt_prob"],
            two_putt_prob,
            shot[f"{team}_team_three_putt_prob"],
        ] + [0] * 7  # Fill the rest with zeros
    else:
        team_probs = []
        for i in range(1, 11):
            prob = shot.get(f"{team}_team_{i}_stroke_prob")
            team_probs.append(prob if pd.notnull(prob) else 0)
    
    return team_probs


def calculate_win_loss_tie_probability(shot: pd.Series):
    """
    Calculate the probability that one team finishes with fewer total strokes than
    the other.

    Args:
        shot (pd.Series): A pandas Series containing the shot-level data for a specific shot.
    
    Returns:
        tuple: A tuple containing the win, loss, and tie probabilities.
    """

    shooting_team_probs = get_probability_vectors(shot, "shooting")
    other_team_probs = get_probability_vectors(shot, "other")

    shooting_team_strokes = shot["shooting_team_strokes"]
    other_team_strokes = shot["other_team_strokes"]

    if shot["shooting_team_ex_strokes"] == 0 and shot["other_team_ex_strokes"] == 0:
        if shooting_team_strokes < other_team_strokes:
            return 1.0, 0.0, 0.0
        elif shooting_team_strokes > other_team_strokes:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 1.0

    if shot["shooting_team_ex_strokes"] == 0:
        win_prob = 0.0
        loss_prob = 0.0
        tie_prob = 0.0
        for other_putts, other_prob in zip(range(1, 11), other_team_probs):
            if other_prob <= 0:
                continue
            other_total = other_team_strokes + other_putts
            if shooting_team_strokes < other_total:
                win_prob += other_prob
            elif shooting_team_strokes > other_total:
                loss_prob += other_prob
            else:
                tie_prob += other_prob
        total_prob = win_prob + loss_prob + tie_prob
        if total_prob > 0:
            win_prob /= total_prob
            loss_prob /= total_prob
            tie_prob /= total_prob
        return win_prob, loss_prob, tie_prob

    if shot["other_team_ex_strokes"] == 0:
        win_prob = 0.0
        loss_prob = 0.0
        tie_prob = 0.0
        for shooting_putts, shooting_prob in  zip(range(1, 11), shooting_team_probs):
            if shooting_prob <= 0:
                continue
            shooting_total = shooting_team_strokes + shooting_putts
            if shooting_total < other_team_strokes:
                win_prob += shooting_prob
            elif shooting_total > other_team_strokes:
                loss_prob += shooting_prob
            else:
                tie_prob += shooting_prob
        total_prob = win_prob + loss_prob + tie_prob
        if total_prob > 0:
            win_prob /= total_prob
            loss_prob /= total_prob
            tie_prob /= total_prob
        return win_prob, loss_prob, tie_prob

    win_prob = 0.0
    loss_prob = 0.0
    tie_prob = 0.0

    for shooting_putts, shooting_prob in  zip(range(1, 11), shooting_team_probs):
        for other_putts, other_prob in  zip(range(1, 11), other_team_probs):
            if shooting_prob <= 0 and other_prob <= 0:
                continue

            shooting_total = shooting_team_strokes + shooting_putts
            other_total = other_team_strokes + other_putts

            if shooting_total < other_total:
                win_prob += shooting_prob * other_prob
            elif shooting_total > other_total:
                loss_prob += shooting_prob * other_prob
            else:
                tie_prob += shooting_prob * other_prob

    total_prob = win_prob + loss_prob + tie_prob
    if total_prob > 0:
        win_prob /= total_prob
        loss_prob /= total_prob
        tie_prob /= total_prob

    return win_prob, loss_prob, tie_prob


def build_hammer_gam(df: pd.DataFrame, target_col: str = "hammer_used_on_hole", feature_cols: List = [], test_size: float = 0.2, random_state: int = 42):
    """
    Fit a LogisticGAM using the requested features, with a
    train/test split and cross-validated hyperparameter tuning.

    Parameters:
        df (pd.DataFrame): DataFrame containing the modeling features and target.
        target_col (str): Column name for the target variable.
        test_size (float): Fraction of rows reserved for the test set.
        random_state (int): Random seed used for reproducibility.

    Returns:
        dict: A dictionary with the trained model and test
              split components.
    """

    if not all(col in df.columns for col in feature_cols + [target_col]):
        missing = [col for col in feature_cols + [target_col] if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    gam = LogisticGAM(s(0, n_splines=15, constraints="monotonic_dec") +
                     te(feature=(1, 2),
                        n_splines=(15, 15),
                        constraints=("monotonic_dec", "monotonic_dec")
                        ), lam=0.6).fit(X_train[feature_cols], y_train)

    return {
        "model": gam,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def analyze_hammer_usage(season: int):
    """
    This function analyzes hammer usage in TGL matches, namely
        - Total hammer usage per match
        - The expected value of a hammer in terms of hole win probability
        - The expected value of accepting/declining a hammer

    Note that this is at the hole level, not match level, so there is no consideration
    of the value of having hammers remaining later in the match

    Args:
        season (int): Season in YYYY format
    """

    # Pull data and isolate to the seson of interest
    hammer_df = sg_data.pull_hammer_data()
    hammer_df = hammer_df[hammer_df["season_year"] == season]

    # Get number of hammers used per hole
    hammer_df["hammers_used"] = hammer_df.groupby(["match_id", "hole_number"])["actionType"].cumcount() + 1
    accept_df = pd.DataFrame(hammer_df.groupby(["match_id", "hole_number"])["actionType"].agg(lambda x: (x == "HAD_HAMMER_ACCEPTED").sum())).reset_index()
    accept_df = accept_df.rename(columns={"actionType": "hammers_accepted"})

    hammer_df = hammer_df.merge(
        accept_df,
        on=["match_id", "hole_number"],
        how="left"
    )

    # Calculate EV of throwing a hammer in a given spot
    # This is the EV after the opposing team chooses to accept/decline
    hammer_df["ev"] = [
        hammers_used * (1 - x) if y == "HAD_HAMMER_DECLINED" else
        (1 + hammers_used) * (x - (1 - x - z)) for x, y, z, hammers_used in zip(hammer_df["win_probability"], hammer_df["actionType"], hammer_df["tie_probability"], hammer_df["hammers_used"])
    ]
    hammer_df["win_ev"] = [
        hammer_dec - no_hammer if y == "HAD_HAMMER_DECLINED" else
        hammer_acc - no_hammer for no_hammer, hammer_dec, hammer_acc, y in zip(
            hammer_df["shooting_win_probability_no_hammer"],
            hammer_df["shooting_win_probability_hammer_declined"],
            hammer_df["shooting_win_probability_hammer_accepted"],
            hammer_df["actionType"]
        )
    ]

    # Should the other team accept?
    hammer_df["accept_ev"] = [
        hammers_used * (1 - x - z) - ((-hammers_used + (1 + hammers_used)*x)/(1 + hammers_used)) for
        x, z, hammers_used in zip(hammer_df["win_probability"], hammer_df["tie_probability"], hammer_df["hammers_used"])
    ]
    hammer_df["accept_win_ev"] = [
        (1 - hammer_acc) - (1 - hammer_dec) for hammer_dec, hammer_acc in zip(
            hammer_df["shooting_win_probability_hammer_declined"],
            hammer_df["shooting_win_probability_hammer_accepted"]
        )
    ]
    hammer_df["decision_ev"] = [
        accept_win_ev if y == "HAD_HAMMER_ACCEPTED" else
        -accept_win_ev for accept_win_ev, y in zip(
            hammer_df["accept_win_ev"],
            hammer_df["actionType"]
        )
    ]
    hammer_df["optimal_decision_ev"] = abs(hammer_df["decision_ev"])

    # Should you throw the hammer?
    hammer_df["throw_ev"] = [
        min(hammer_acc, hammer_dec) - no_hammer for no_hammer, hammer_dec, hammer_acc in zip(
            hammer_df["shooting_win_probability_no_hammer"],
            hammer_df["shooting_win_probability_hammer_declined"],
            hammer_df["shooting_win_probability_hammer_accepted"],
        )
    ]

    # Hole value
    hammer_df["hole_value"] = [
        0 if pd.isnull(winning_team_id) else
        hammers_used if pd.notnull(winning_team_id) and action_type == "HAD_HAMMER_DECLINED" else
        hammers_accepted + 1 for winning_team_id, hammers_used, action_type, hammers_accepted in zip(
            hammer_df["winning_team_id"],
            hammer_df["hammers_used"],
            hammer_df["actionType"],
            hammer_df["hammers_accepted"]
        )
    ]

    hammer_df["realized_value"] = [
        0 if pd.isnull(winning_team_id) else
        hole_value if team_id == winning_team_id else
        -hole_value for winning_team_id, hole_value, team_id in zip(hammer_df["winning_team_id"], hammer_df["hole_value"], hammer_df["teamId"])
    ]

    hammer_df["other_team"] = [
        winning_team if winning_team != team else 
        losing_team for winning_team, losing_team, team in zip(hammer_df["winning_team_id"],
                                                               hammer_df["losing_team_id"],
                                                               hammer_df["teamId"])
    ]

    return hammer_df
