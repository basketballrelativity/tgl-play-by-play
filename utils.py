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


def process_shot_data(shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the shot-level data to derive expected strokes
    for drives, putts, and approach shots.

    Args:
        shot_df (pd.DataFrame): A DataFrame containing the shot-level data.
    
    Returns:
        pd.DataFrame: A DataFrame with expected strokes for drives, putts, and approach shots.
    """

    def next_stroke_func(x):
        """quick-and-dirty util to get next ex strokes"""
        return x["approach_ex_strokes"] if pd.notnull(x["approach_ex_strokes"]) else \
                0 if str(x["shot_location"]) == "Hole" or str(x["strokeType"])=="GIMME" else \
                x["putt_ex_strokes"]

    def next_putt_func(x, putt_type):
        """quick-and-dirty util to get next putt probs"""
        return np.nan if pd.notnull(x["approach_ex_strokes"]) else \
                1 if str(x["shot_location"]) == "Hole" or str(x["strokeType"]) == "GIMME" else \
                x[putt_type]
    

    # Apply ex strokes models
    shot_df = sg_utils.get_drive_ex_strokes(shot_df)
    shot_df = sg_utils.get_putt_ex_strokes(shot_df)
    shot_df = sg_utils.get_approach_ex_strokes(shot_df)

    # Grab holes and initialize DataFrame
    unique_holes = list(set(shot_df["hole_number"]))
    strokes_df = pd.DataFrame()

    # This loop is a bit confusing, but because the end_distance value on
    # a hole is used as a feature for the approach and putting shots, we must
    # do the following:
    # - For the first shot by each team, we automatically pass the length of the hole
    #   through the drive ex strokes model and use those probabilities
    # - For every other shot, we take the location and end_distance of the prior shot
    #   (denoted as the "next" variables below) and use those to determine whether we should use
    #   the approach or putting ex strokes predictions for each subsequent shot
    for hole in unique_holes:
        # Isolate to hole and order by time
        hole_shots = shot_df[shot_df["hole_number"] == hole]
        hole_shots = hole_shots.sort_values("shot_number", ascending=True)
        unique_teams = list(set(hole_shots["teamId"]))
        for team in unique_teams:
            # Isolate to team and initialize storage
            team_shots = hole_shots[(hole_shots["teamId"] == team)]
            ex_strokes = []
            one_putt_prob = []
            three_putt_prob = []
            shot_number = []
            stroke_prob_rows = []
            for _, shot in team_shots.iterrows():
                shot_number.append(shot["shot_number"])
                # If this is our first shot, we're using drive ex strokes
                if pd.notnull(shot["shot_number"]) and shot["shot_number"] == 1:
                    # Grab drive and approach probability columns
                    drive_prob_cols = list(dict.fromkeys(
                        col for col in shot_df.columns if "drive_stroke_prob" in col
                    ))
                    app_prob_cols = list(dict.fromkeys(
                        col for col in shot_df.columns if "app_stroke_prob" in col
                    ))

                    # Store drive ex strokes
                    ex_strokes.append(shot["drive_ex_strokes"])
                    one_putt_prob.append(np.nan)
                    three_putt_prob.append(np.nan)
                    
                    # If approach_ex_strokes exists, we know the next shot is on approach
                    # If not, look for key words designating a hole out. Otherwise, we're puttin'
                    next_stroke = next_stroke_func(shot)
                    
                    # Store putts if we're puttin'
                    next_one_putt = next_putt_func(shot, "one_putt")
                    next_three_putt = next_putt_func(shot, "three_putt")

                    # Use the drive columns as our shot-type agnostic probability columns
                    # Store the approach probabilities as our next probability columns
                    # Note this doesn't get used if we're puttin'
                    stroke_prob_rows.append({
                        col.replace("drive_", ""): shot[col]
                        for col in drive_prob_cols
                    })
                    next_probs = {
                        col.replace("app_", ""): shot[col]
                        for col in app_prob_cols
                    }
                else:
                    # Store our values of interest that we defined above
                    # or here from the last shot
                    ex_strokes.append(next_stroke)
                    one_putt_prob.append(next_one_putt)
                    three_putt_prob.append(next_three_putt)
                    stroke_prob_rows.append(next_probs)

                    # If this isn't a penalty stroke, define the relevant values for the next shot
                    # If it's a penalty stroke, we just reuse the prior values
                    if shot["strokeType"] != "PENALTY":
                        next_stroke = next_stroke_func(shot)
                        next_one_putt = next_putt_func(shot, "one_putt")
                        next_three_putt = next_putt_func(shot, "three_putt")
                        next_probs = {
                            col.replace("app_", ""): shot[col]
                            for col in app_prob_cols
                        }

            # Convert stroke probabilities to a DataFrame
            prob_df = pd.DataFrame(stroke_prob_rows)

            # Store ex strokes and putting probabilities
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

            # Copy over probabilities
            for col in prob_df.columns:
                team_hole_df[col] = prob_df[col].values
            strokes_df = pd.concat([strokes_df, team_hole_df])

    # Remove columns from shot_df since we'll be overwriting those
    # based on the logic detailed above at the start of the loop
    for col in prob_df.columns:
        if col in shot_df.columns:
            del shot_df[col]
    
    # Maintain a left join in case we missed anything
    shot_df = shot_df.merge(
        strokes_df,
        on=["teamId", "hole_number", "shot_number"],
        how="left"
    )

    return shot_df


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
