"""
This script contains functions to simulate
a match given shot-level win probabilities
and information on the holes remaining
"""
import pickle

import numpy as np
import pandas as pd

import sg_data
import utils

def load_hammer_probability_model():
    """ This function loads and returns the hammer
    probability model (function of score, holes remaining, and
    hammers used)

    Returns:

        model_dict (dict): Dictionary containing model objects for
            the hammer probability model
    """

    with open('hammer_model.pkl', 'rb') as handle:
        model_dict = pickle.load(handle)

    return model_dict


def simulate_match(shot: pd.Series, hole_df: pd.DataFrame, hammer_df: pd.DataFrame, hole_value: int, sims: int = 1000):
    """ This function simulates a match based on shot-level
    win probabilities and the holes remaining

    Args:
        shot (pd.Series): Series containing shot-level information
        hole_df (pd.DataFrame): DataFrame containing information on all
            holes in a match
        hammer_df (pd.DataFrame): DataFrame containing information on the number of
            hammers remaining
        hole_value (int): Value of the current hole
        sims (int): Number of match simulations to conduct

    Returns:
        tuple: A tuple containing the win and loss probabilities, along with the score differential
    """

    # Get current hole
    current_hole = shot["hole_number"]
    team_id = shot["shooting_team"]
    score_dict = {}

    # Outcomes
    base_outcomes = [-1, 0, 1]
    outcomes = [outcome * hole_value for outcome in base_outcomes]
    probs = [shot["loss_probability"], shot["tie_probability"], shot["win_probability"]]

    shot_samples = np.random.choice(outcomes, size=sims, p=probs)
    score_dict[current_hole] = shot_samples

    # Rest of the holes
    hole_df = hole_df[hole_df["hole_number"] > current_hole]

    # Pull in hammer information (acore_diff is prior to playing the hole of interest)
    score_diff = hammer_df[(hammer_df["teamId"]==team_id) & (hammer_df["hole_number"]==current_hole)]["score_diff"].iloc[0]

    # Get ex strokes and probabilities off the tee for the remaining holes
    if current_hole < 15:
        hole_df = utils.get_drive_ex_strokes(hole_df).sort_values("hole_number")
        rename_dict = {}
        for col in hole_df.columns:
            if "_drive_stroke_prob" in col:
                split_col = col.split("_")[0]
                rename_dict[col] = "shooting_team_" + split_col + "_stroke_prob"
                hole_df["other_team_" + split_col + "_stroke_prob"] = list(hole_df[col])

        rename_dict["drive_ex_strokes"] = "shooting_team_ex_strokes"
        hole_df["other_team_ex_strokes"] = list(hole_df["drive_ex_strokes"])
        hole_df = hole_df.rename(columns=rename_dict)
    
        hole_df["shooting_team_strokes"] = 0
        hole_df["other_team_strokes"] = 0
        hole_df["shooting_team_one_putt_prob"] = np.nan
        hole_df["other_team_one_putt_prob"] = np.nan

        # Load hammer probability model and hammer data
        model_dict = load_hammer_probability_model()
        hammer_df = hammer_df[(hammer_df["match_id"]==shot["match_id"]) & (hammer_df["hole_number"] > current_hole)]

        # Loop through holes
        for _, hole in hole_df.iterrows():
            win_prob, loss_prob, tie_prob = utils.calculate_win_loss_tie_probability(hole)

            probs = [win_prob, tie_prob, loss_prob]
            hole_samples = np.random.choice(base_outcomes, size=sims, p=probs)
            score_dict[hole["hole_number"]] = hole_samples

    score_df = pd.DataFrame(score_dict)
    score_df["total"] = score_df.sum(axis=1) + score_diff

    # Split ties 50-50
    tie_prob = len(score_df[score_df["total"]==0])/len(score_df)
    win_prob = 0.5*tie_prob + len(score_df[score_df["total"]>0])/len(score_df)
    loss_prob = 0.5*tie_prob + len(score_df[score_df["total"]<0])/len(score_df)

    return win_prob, loss_prob, score_diff
