"""
This script contains functionality
to run inference using the off-the-tee,
approach, and putting expected strokes
models
"""

import pickle

import pandas as pd

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
