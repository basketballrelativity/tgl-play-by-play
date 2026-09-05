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
from scipy.stats import skellam, poisson
from scipy.special import expit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.calibration import calibration_curve
from statsmodels.miscmodels.ordinal_model import OrderedModel, OrderedResults

import matplotlib.pyplot as plt


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
    with open('drive_models.pkl', 'rb') as file:
        drive_models = pickle.load(file)

    # Filter for drive shots (assuming drive shots are the first shot of each hole)
    par_dict = {}
    alt_shot_df = pd.DataFrame()
    for par in [3, 4, 5]:
        par_df = shot_df[shot_df["hole_par"] == par]
        par_df = par_df.rename({"yards" : "distance"}, axis=1)
        preds = drive_models[f"par_{par}"].predict(par_df[["distance"]])
        if par < 5:
            strokes = np.array(range(1, len(preds.columns)+1))
        else:
            strokes = np.array(range(2, len(preds.columns)+2))
        par_df["drive_ex_strokes"] = np.dot(preds.values, strokes)
        for stroke in strokes:
            if par < 5:
                par_df[f"{stroke}_drive_stroke_prob"] = preds[stroke-1]
            else:
                par_df[f"{stroke}_drive_stroke_prob"] = preds[stroke-2]
        par_dict[par] = par_df.copy()
    
    for par in [3, 4, 5]:
        alt_shot_df = pd.concat([alt_shot_df, par_dict[par]])

    return alt_shot_df


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
                    next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else shot["putt_ex_strokes"]
                    next_one_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else shot["one_putt"]
                    next_three_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else shot["three_putt"]
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
                    next_stroke = shot["approach_ex_strokes"] if pd.notnull(shot["approach_ex_strokes"]) else shot["putt_ex_strokes"]
                    next_one_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else shot["one_putt"]
                    next_three_putt = np.nan if pd.notnull(shot["approach_ex_strokes"]) else shot["three_putt"]
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


def prob_x_greater_than_y_skellam(stroke_diff, mu1, mu2):
    # Z = X - Y follows a Skellam distribution with parameters mu1 and mu2
    # P(X > Y) = P(X - Y >= 1) = 1 - P(X - Y <= 0)
    # Probability that x > y is the probability that y wins the hole
    return 1 - skellam.cdf(stroke_diff, mu1, mu2)


def poisson_greater_than_x(x: int, mu: float) -> float:
    """
    Calculates the probability that a Poisson random variable 
    is strictly greater than x.
    
    Parameters:
    x (int): The threshold value (non-negative integer).
    mu (float): The mean rate (lambda) of the distribution.
    
    Returns:
    float: Probability P(X > x)
    """
    # sf(x) handles the calculation as 1 - cdf(x) internally with high precision
    return poisson.sf(x, mu)


def poisson_less_than_x(x, mu):
    """
    Calculates the probability that a Poisson variable is strictly less than x.
    P(X < x) = P(X <= x - 1)
    
    Parameters:
    x (int): The upper limit (exclusive)
    mu (float): The expected mean rate (lambda) of the distribution
    
    Returns:
    float: The cumulative probability
    """
    # If x is 0 or negative, probability of being strictly less than it is 0
    if x <= 0:
        return 0.0
        
    return poisson.cdf(x - 1, mu)


def calculate_on_green_fewer_strokes_probability(shot: pd.Series):
    """
    Calculate the probability that one team finishes with fewer total strokes than
    the other when both teams are on the green.

    A team can finish in 1, 2, or 3 putts, with probabilities derived from the
    corresponding one-putt and three-putt probabilities. If either team has
    ex_strokes == 0, that team has already holed out and contributes a fixed final
    total with no remaining putt distribution.
    """
    shooting_team_one_putt_prob = shot.get("shooting_team_one_putt_prob")
    other_team_one_putt_prob = shot.get("other_team_one_putt_prob")
    shooting_team_ex_strokes = shot.get("shooting_team_ex_strokes")
    other_team_ex_strokes = shot.get("other_team_ex_strokes")

    if pd.isnull(shooting_team_one_putt_prob) or pd.isnull(other_team_one_putt_prob):
        return np.nan, np.nan, np.nan

    shooting_team_three_putt_prob = shot.get("shooting_team_three_putt_prob", 0)
    other_team_three_putt_prob = shot.get("other_team_three_putt_prob", 0)

    shooting_team_one_putt_prob = float(shooting_team_one_putt_prob)
    other_team_one_putt_prob = float(other_team_one_putt_prob)
    shooting_team_three_putt_prob = float(shooting_team_three_putt_prob)
    other_team_three_putt_prob = float(other_team_three_putt_prob)

    shooting_team_two_putt_prob = max(0.0, 1.0 - shooting_team_one_putt_prob - shooting_team_three_putt_prob)
    other_team_two_putt_prob = max(0.0, 1.0 - other_team_one_putt_prob - other_team_three_putt_prob)

    outcomes = {
        "shooting": {1: shooting_team_one_putt_prob, 2: shooting_team_two_putt_prob, 3: shooting_team_three_putt_prob},
        "other": {1: other_team_one_putt_prob, 2: other_team_two_putt_prob, 3: other_team_three_putt_prob},
    }

    shooting_team_strokes = shot["shooting_team_strokes"]
    other_team_strokes = shot["other_team_strokes"]

    if shooting_team_ex_strokes == 0 and other_team_ex_strokes == 0:
        if shooting_team_strokes < other_team_strokes:
            return 1.0, 0.0, 0.0
        elif shooting_team_strokes > other_team_strokes:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 1.0

    if shooting_team_ex_strokes == 0:
        win_prob = 0.0
        loss_prob = 0.0
        tie_prob = 0.0
        for other_putts, other_prob in outcomes["other"].items():
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

    if other_team_ex_strokes == 0:
        win_prob = 0.0
        loss_prob = 0.0
        tie_prob = 0.0
        for shooting_putts, shooting_prob in outcomes["shooting"].items():
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

    for shooting_putts, shooting_prob in outcomes["shooting"].items():
        for other_putts, other_prob in outcomes["other"].items():
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


def calculate_mixed_green_probability(shot: pd.Series, on_green_team: str = "shooting"):
    """
    Calculate win/loss/tie probabilities when one team is on the green and the
    other is not.

    The off-green team is modeled with a Poisson distribution using ex_strokes as
    the rate parameter. The on-green team is modeled with a discrete 1/2/3 putt
    distribution built from one_putt_prob and three_putt_prob. If either team has
    ex_strokes == 0, it has already holed out and contributes a fixed final total
    rather than additional probability mass.

    Returns probabilities for the shooting team in the same convention as the
    rest of this module.
    """
    if on_green_team == "shooting":
        on_green_one_putt_prob = shot.get("shooting_team_one_putt_prob")
        on_green_three_putt_prob = shot.get("shooting_team_three_putt_prob", 0)
        on_green_ex_strokes = shot.get("shooting_team_ex_strokes")
        off_green_rate = shot.get("other_team_ex_strokes")
        on_green_strokes = shot.get("shooting_team_strokes")
        off_green_strokes = shot.get("other_team_strokes")
    elif on_green_team == "other":
        on_green_one_putt_prob = shot.get("other_team_one_putt_prob")
        on_green_three_putt_prob = shot.get("other_team_three_putt_prob", 0)
        on_green_ex_strokes = shot.get("other_team_ex_strokes")
        off_green_rate = shot.get("shooting_team_ex_strokes")
        on_green_strokes = shot.get("other_team_strokes")
        off_green_strokes = shot.get("shooting_team_strokes")
    else:
        raise ValueError("on_green_team must be either 'shooting' or 'other'")

    if pd.isnull(on_green_one_putt_prob) and pd.isnull(off_green_rate):
        return np.nan, np.nan, np.nan

    on_green_one_putt_prob = float(on_green_one_putt_prob)
    on_green_three_putt_prob = float(on_green_three_putt_prob)
    off_green_rate = float(off_green_rate)
    on_green_two_putt_prob = max(0.0, 1.0 - on_green_one_putt_prob - on_green_three_putt_prob)

    if on_green_ex_strokes == 0 and off_green_rate == 0:
        if on_green_strokes < off_green_strokes:
            win_prob, loss_prob, tie_prob = (1.0, 0.0, 0.0)
        elif on_green_strokes > off_green_strokes:
            win_prob, loss_prob, tie_prob = (0.0, 1.0, 0.0)
        else:
            win_prob, loss_prob, tie_prob = (0.0, 0.0, 1.0)
        if on_green_team == "other":
            return loss_prob, win_prob, tie_prob
        return win_prob, loss_prob, tie_prob

    if on_green_ex_strokes == 0:
        fixed_total = float(on_green_strokes)
        diff = fixed_total - float(off_green_strokes)

        if diff < 0:
            win_prob = 0.0
            loss_prob = 1.0
            tie_prob = 0.0
        elif diff == 0:
            win_prob = 0.0
            tie_prob = poisson.pmf(0, off_green_rate)
            loss_prob = 1.0 - tie_prob
        else:
            win_prob = poisson.sf(diff, off_green_rate)
            tie_prob = poisson.pmf(diff, off_green_rate)
            loss_prob = 1.0 - win_prob - tie_prob

        if on_green_team == "other":
            return loss_prob, win_prob, tie_prob
        return win_prob, loss_prob, tie_prob

    if off_green_rate == 0:
        win_prob = 0.0
        loss_prob = 0.0
        tie_prob = 0.0
        for putts, p in [(1, on_green_one_putt_prob), (2, on_green_two_putt_prob), (3, on_green_three_putt_prob)]:
            if p <= 0:
                continue
            on_green_total = on_green_strokes + putts
            if on_green_total < off_green_strokes:
                win_prob += p
            elif on_green_total > off_green_strokes:
                loss_prob += p
            else:
                tie_prob += p
        total_prob = win_prob + loss_prob + tie_prob
        if total_prob > 0:
            win_prob /= total_prob
            loss_prob /= total_prob
            tie_prob /= total_prob
        if on_green_team == "other":
            return loss_prob, win_prob, tie_prob
        return win_prob, loss_prob, tie_prob

    win_prob = 0.0
    loss_prob = 0.0
    tie_prob = 0.0
    diff = on_green_strokes - off_green_strokes

    for putts, p in [(1, on_green_one_putt_prob), (2, on_green_two_putt_prob), (3, on_green_three_putt_prob)]:
        if p <= 0:
            continue

        threshold = putts + diff
        win_prob += p * poisson.sf(threshold, off_green_rate - 1)
        tie_prob += p * poisson.pmf(threshold, off_green_rate - 1)
        loss_prob += p * (1.0 - poisson.sf(threshold, off_green_rate - 1) - poisson.pmf(threshold, off_green_rate - 1))

    total_prob = win_prob + loss_prob + tie_prob
    if total_prob > 0:
        win_prob /= total_prob
        loss_prob /= total_prob
        tie_prob /= total_prob

    if on_green_team == "other":
        return loss_prob, win_prob, tie_prob

    return win_prob, loss_prob, tie_prob


def calculate_win_loss_tie_probability(shot: pd.Series) -> pd.Series:
    """
    Calculate the win, loss, and tie probabilities for a given shot.

    Args:
        shot (pd.Series): A Series containing the shot data.
    
    Returns:
        tuple: A tuple containing the win, loss, and tie probabilities.
    """
    shooting_team_ex_strokes = shot["shooting_team_ex_strokes"]
    other_team_ex_strokes = shot["other_team_ex_strokes"]

    shooting_team_strokes = shot["shooting_team_strokes"]
    other_team_strokes = shot["other_team_strokes"]

    # Calculate the probability of winning, losing, and tying
    if shooting_team_ex_strokes > 0 and other_team_ex_strokes > 0 and pd.isnull(shot["shooting_team_one_putt_prob"]) and pd.isnull(shot["other_team_one_putt_prob"]):
        win_prob = prob_x_greater_than_y_skellam(
            shooting_team_strokes - other_team_strokes,
            other_team_ex_strokes - 1,
            shooting_team_ex_strokes - 1
        )
        loss_prob = prob_x_greater_than_y_skellam(
            other_team_strokes - shooting_team_strokes,
            shooting_team_ex_strokes - 1,
            other_team_ex_strokes - 1
        )
        tie_prob = 1 - win_prob - loss_prob
    elif pd.notnull(shot["shooting_team_one_putt_prob"]) and pd.isnull(shot["other_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_mixed_green_probability(shot, on_green_team="shooting")
    elif pd.notnull(shot["other_team_one_putt_prob"]) and pd.isnull(shot["shooting_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_mixed_green_probability(shot, on_green_team="other")
    elif pd.notnull(shot["shooting_team_one_putt_prob"]) and pd.notnull(shot["other_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_on_green_fewer_strokes_probability(shot)
    elif shooting_team_ex_strokes == 0 and pd.notnull(shot["other_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_on_green_fewer_strokes_probability(shot)
    elif other_team_ex_strokes == 0 and pd.notnull(shot["shooting_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_on_green_fewer_strokes_probability(shot)
    elif shooting_team_ex_strokes == 0 and pd.isnull(shot["other_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_mixed_green_probability(shot, on_green_team="shooting")
    elif other_team_ex_strokes == 0 and pd.isnull(shot["shooting_team_one_putt_prob"]):
        win_prob, loss_prob, tie_prob = calculate_mixed_green_probability(shot, on_green_team="other")
    else:
        win_prob, loss_prob, tie_prob = np.nan, np.nan, np.nan

    return win_prob, loss_prob, tie_prob


def build_gradient_boosting_model(df: pd.DataFrame, target_col: str = "result", test_size: float = 0.2, random_state: int = 42):
    """
    Fit a GradientBoostingClassifier using the requested features, with a
    train/test split and cross-validated hyperparameter tuning.

    Parameters:
        df (pd.DataFrame): DataFrame containing the modeling features and target.
        target_col (str): Column name for the target variable.
        test_size (float): Fraction of rows reserved for the test set.
        random_state (int): Random seed used for reproducibility.

    Returns:
        dict: A dictionary with the trained pipeline, best parameters, and test
              split components.
    """
    feature_cols = [
        "shooting_team_ex_strokes",
        "shooting_team_strokes",
        "strokes_diff",
        "ex_strokes_diff",
        "hole_par",
        "shot_number",
    ]

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

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )

    param_grid = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [4, 5, 6],
        "model__min_samples_leaf": [100, 150, 200],
    }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return {
        "pipeline": grid,
        "best_params_": grid.best_params_,
        "best_score_": grid.best_score_,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def visualize_calibration(data_df, result_type="win"):
    """ This function visualizes calibration for the
    hole win probability model

    @param data_df (DataFrame): DataFrame containing
        win, loss, and tie indicators and game time remaining
    @param result_type (str): Type of result to visualize. Options are "win" or "tie".

    Returns:

        fig (plt.figure): Figure object of the win probability
            visualization
    """

    # Narrow to valid predictions
    data_df = data_df[pd.notnull(data_df[result_type + "_probability"])]
    data_df = data_df[data_df["shot_number"] > 1]
 
    prob_true, prob_pred = calibration_curve(data_df[result_type],
                                             data_df[result_type + "_probability"], n_bins=10)

    fig = plt.figure(0, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax1.plot(prob_pred, prob_true, "s-",)

    ax2.hist(data_df[result_type + "_probability"], range=(0, 1), bins=10,
            histtype="step", lw=2)

    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title(f"Hole {result_type.title()} Probability Calibration", fontsize=16)
    ax2.set_xlabel("Predicted Probability", fontsize=14)
    ax1.set_ylabel("Actual Probability", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f"hole_{result_type}_probability_calibration.png")
    plt.close(fig)