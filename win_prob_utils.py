"""
This script contains functions to simulate
a match given shot-level win probabilities
and information on the holes remaining
"""
import pickle

import numpy as np
import pandas as pd

from hammer_model import FEATURES
import utils

SEASON = 2026

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


def get_hammer_value():
    """ This function pulls in the probability of
    different hole outcomes conditioned on a team
    deploying the hammer. This ignores both teams
    deploying their hammer on the same hole given
    its rarity and how the simulation is set up

    Returns:

        - value_df: DataFrame of the probability of
            a hammer being worth a certain point level
    """

    hammer_df = utils.analyze_hammer_usage(SEASON)
    hammer_df["realized_value"] = [max(-2, min(2, x)) for x in hammer_df["realized_value"]]

    value_df = pd.DataFrame(
        hammer_df.groupby(["realized_value"]
    )["match_id"].count()/len(hammer_df)).reset_index().sort_values("realized_value")

    value_df = value_df.rename(columns={"match_id": "probs"})

    return value_df


def get_hammer_deployment_probabilities(hole: pd.Series, score_dict: dict, score_diff: int, hammers: pd.DataFrame, other_hammers: pd.DataFrame, model_dict: dict, sims: int):
    """ This function deploys the hammer use probability model
    on future holes and overrides the sampling of the pre-hole
    win/loss/tie probabilities

    Args:
        hole (pd.Series): Series containing hole information
        score_dict (dict): Dictionary containing simulated scores
            on a hole
        score_diff (int): Score differential relative to the team
            of interest
        hammers (pd.DataFrame): Hammers used by the team of interest
        other_hammes (pd.DataFrame): Hammers used by the other team

    Returns:
        hammer_df (pd.DataFrame): DataFrame containing hammer
            deployment probabilities
    """

    if len(score_dict) == 0:
        scores = [score_diff]*sims
    else:
        scores = pd.DataFrame(score_dict).sum(axis=1) + score_diff

    # Calculate holes remaining
    holes_remaining = 15 - hole["hole_number"] + 1

    hammer_df = pd.DataFrame(
        {
            "score_diff": list(scores),
            "holes_remaining_prior": [holes_remaining]*len(scores),
            "hammers_used_prior": list(hammers)
        }
    )

    hammer_df["hammer_probability"] = model_dict["model"].predict_proba(hammer_df[FEATURES])
    hammer_df["hammer_probability"] = [0 if hammer_count >= 3 else x for hammer_count, x in zip(hammer_df["hammers_used_prior"], hammer_df["hammer_probability"])]

    # Now for the other team
    hammer_df["score_diff"] = -hammer_df["score_diff"]
    hammer_df["hammers_used_prior"] = list(other_hammers)
    hammer_df["other_hammer_probability"] = model_dict["model"].predict_proba(hammer_df[FEATURES])
    hammer_df["other_hammer_probability"] = [0 if hammer_count >= 3 else x for hammer_count, x in zip(hammer_df["hammers_used_prior"], hammer_df["other_hammer_probability"])]

    return hammer_df


def simulate_match(
        shot: pd.Series,
        hole_df: pd.DataFrame,
        value_df: pd.DataFrame,
        hammers_used: int,
        other_hammers_used: int,
        hole_value: int,
        score_diff: int,
        current_hole: int,
        complete_hole: bool = True,
        sims: int = 1000
    ):
    """ This function simulates a match based on shot-level
    win probabilities and the holes remaining

    Args:
        shot (pd.Series): Series containing shot-level information
        hole_df (pd.DataFrame): DataFrame containing information on all
            holes in a match
        value_df (pd.DataFrame): Probabilities of hole-level point outcomes conditioned on
            a hammer being deployed
        hammers_used (int): Number of hammers remaining for the team of interest
        other_hammers_used (int): Number of hammers remaining for the other team
        hole_value (int): Value of the current hole
        score_diff (int): Score of the match relative to the shooting team
        sims (int): Number of match simulations to conduct

    Returns:
        tuple: A tuple containing the win and loss probabilities, along with the score differential
    """

    # Get current hole
    score_dict = {}
    base_outcomes = [-1, 0, 1]

    # Outcomes
    if complete_hole:
        outcomes = [outcome * hole_value for outcome in base_outcomes]
        probs = [shot["loss_probability"], shot["tie_probability"], shot["win_probability"]]

        shot_samples = np.random.choice(outcomes, size=sims, p=probs)
        score_dict[current_hole] = shot_samples

        # Rest of the holes
        hole_df = hole_df[hole_df["hole_number"] > current_hole]
    else:
        # Sim from start of the current hole (used to derive hammer value)
        hole_df = hole_df[hole_df["hole_number"] >= current_hole]

    # Get ex strokes and probabilities off the tee for the remaining holes
    if (current_hole < 15) or (not complete_hole):
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

        # Fill in hammer_dict
        hammer_dict = {}
        other_hammer_dict = {}

        # Negative one is a placeholder here so as to not overwrite the current
        # hole hammers used when calculating hammer value
        hammer_dict[-1] = [hammers_used]*sims
        other_hammer_dict[-1] = [other_hammers_used]*sims

        # Loop through holes
        for _, hole in hole_df.iterrows():
            hammers = pd.DataFrame(hammer_dict).sum(axis=1)
            other_hammers = pd.DataFrame(other_hammer_dict).sum(axis=1)

            hammer_prob_df = get_hammer_deployment_probabilities(
                hole, score_dict, score_diff, hammers, other_hammers, model_dict, sims
            )
            hammer_samples = np.random.binomial(n=1, p=hammer_prob_df["hammer_probability"])
            other_hammer_samples = np.random.binomial(n=1, p=hammer_prob_df["other_hammer_probability"])

            mask = []
            hammer_sample_list = []
            hammer_used_hole = []
            other_hammer_used_hole = []

            for h_sample, oh_sample, h_used, oh_used in zip(hammer_samples, other_hammer_samples, hammers, other_hammers):
                if (h_sample == 1 and h_used < 3) and (oh_sample == 1 and oh_used < 3):
                    mask.append(True)
                    # Randomize which team actually uses the hammer
                    binom_sample = np.random.binomial(n=1, p=0.5)
                    if binom_sample == 1:
                        hammer_outcomes = list(value_df["realized_value"])
                        hammer_used_hole.append(1)
                        other_hammer_used_hole.append(0)
                    else:
                        hammer_outcomes = list(-value_df["realized_value"])
                        hammer_used_hole.append(0)
                        other_hammer_used_hole.append(1)
                
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])
                elif h_sample == 1 and h_used < 3:
                    mask.append(True)
                    hammer_outcomes = list(value_df["realized_value"])
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])

                    hammer_used_hole.append(1)
                    other_hammer_used_hole.append(0)
                elif oh_sample == 1 and oh_used < 3:
                    mask.append(True)
                    hammer_outcomes = list(-value_df["realized_value"])
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])
                    hammer_used_hole.append(0)
                    other_hammer_used_hole.append(1)
                else:
                    mask.append(False)
                    hammer_used_hole.append(0)
                    other_hammer_used_hole.append(0)

            win_prob, loss_prob, tie_prob = utils.calculate_win_loss_tie_probability(hole)

            probs = [win_prob, tie_prob, loss_prob]
            hole_samples = np.random.choice(base_outcomes, size=sims, p=probs)

            hole_samples[mask] = hammer_sample_list
            score_dict[hole["hole_number"]] = hole_samples
            hammer_dict[hole["hole_number"]] = hammer_used_hole
            other_hammer_dict[hole["hole_number"]] = other_hammer_used_hole

    score_df = pd.DataFrame(score_dict)
    score_df["total"] = score_df.sum(axis=1) + score_diff

    # Split ties 50-50
    tie_prob = len(score_df[score_df["total"]==0])/len(score_df)
    win_prob = 0.5*tie_prob + len(score_df[score_df["total"]>0])/len(score_df)
    loss_prob = 0.5*tie_prob + len(score_df[score_df["total"]<0])/len(score_df)

    return win_prob, loss_prob, score_diff
