"""
This script contains functionality
to run inference using the off-the-tee,
approach, and putting expected strokes
models
"""

import pickle

import pandas as pd
import numpy as np
from scipy.special import expit


# Possible hole par values
HOLE_PARS = [3, 4, 5]


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
    sg_df = pd.DataFrame()
    for par in HOLE_PARS:
        par_df = shot_df[shot_df["hole_par"] == par]
        if len(par_df) > 0:
            # Set initial stroke value (no one holes out on
            # a par 5, in this dataset at least)
            if par == 5:
                offset = 2
            else:
                offset = 1

            # Rename yards
            par_df = par_df.rename({"yards" : "distance"}, axis=1)
            par_df["distance"] = pd.to_numeric(par_df["distance"])

            # Predict
            preds = drive_models[f"par_{par}"].predict(par_df[["distance"]])
            strokes = np.array(range(offset, len(preds.columns)+offset))

            # Expected strokes
            par_df["drive_ex_strokes"] = np.dot(preds.values, strokes)
            for stroke in strokes:
                par_df[f"{stroke}_drive_stroke_prob"] = preds[stroke-offset]
        
        sg_df = pd.concat([sg_df, par_df.copy()])
    
    return sg_df


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

        return expit(log_odds)[0] 
        

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

    # Filter to possible approach shots
    approach_df = shot_df[
        (shot_df["shot_location"].isin(["Fairway", "Rough", "Bunker", "Native Area", "Free Drop Area", "Penalty Area"])) &
        (shot_df["strokeType"] == "SHOT")
    ]

    # Engineer and scale features
    features = ["distance_norm", "fairway", "rough", "bunker", "native_area", "other"]
    approach_df["distance_norm"] = approach_model["distance_scaler"].transform(np.array(approach_df["end_distance"]).reshape(-1, 1))
    approach_df["fairway"] = [1 if loc in ["Fairway", "Free Drop Area"] else 0 for loc in approach_df["shot_location"]]
    approach_df["rough"] = [1 if loc == "Rough" else 0 for loc in approach_df["shot_location"]]
    approach_df["bunker"] = [1 if loc == "Bunker" else 0 for loc in approach_df["shot_location"]]
    approach_df["native_area"] = [1 if loc == "Native Area" else 0 for loc in approach_df["shot_location"]]
    approach_df["other"] = [1 if loc not in ["Fairway", "Rough", "Bunker", "Native Area", "Free Drop Area"] else 0 for loc in approach_df["shot_location"]]

    # Recall, the approach model is actually a set of two models
    # This first model predicts ex strokes via a Poisson regression
    # But, because the Poisson distribution doesn't capture the distribution
    # of expected strokes well, a second ordered regression model predicts the probability
    # of finishing the hole in a given number of strokes
    approach_df["approach_ex_strokes"] = approach_model["ex_strokes_model"].predict(approach_df[features])

    # Apply the ordered regression model
    preds = approach_model["prob_model"].predict(approach_df[["approach_ex_strokes", "fairway", "bunker", "rough"]])
    strokes = np.array(range(1, len(preds.columns)+1))
    for stroke in strokes:
        approach_df[f"{stroke}_app_stroke_prob"] = preds[stroke-1]

    # Join onto the original shot DataFrame
    shot_df = shot_df.merge(
        approach_df[
        ["match_id", "hole_config_id", "hole_id", "hole_number",
         "sequence", "shot_number", "playerId", "teamId", "approach_ex_strokes"] + \
         [col for col in approach_df.columns if col.endswith('_app_stroke_prob')]
        ],
        on=[
            "match_id", "hole_config_id", "hole_id", "hole_number",
            "sequence", "shot_number", "playerId", "teamId"
            ],
        how="left"
    )

    return shot_df