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
import constants


def load_hammer_probability_model():
    """ This function loads and returns the hammer
    probability model (function of score, holes remaining, and
    hammers used)

    Returns:

        model_dict (dict): Dictionary containing model objects for
            the hammer probability model
    """

    with open('models/hammer_model.pkl', 'rb') as handle:
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

    # This ignores three-point holes where both teams use the hammer
    hammer_df = utils.analyze_hammer_usage(constants.SEASON)
    hammer_df["realized_value"] = [max(-2, min(2, x)) for x in hammer_df["realized_value"]]

    # Probability of the throwing team realizing
    # -2, -1, 0, 1, or 2 points
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
        sims (int): Number of match simulations to conduct

    Returns:
        hammer_df (pd.DataFrame): DataFrame containing hammer
            deployment probabilities
    """

    # Need to initialize scores if we haven't simmed holes yet
    if len(score_dict) == 0:
        scores = [score_diff]*sims
    else:
        scores = pd.DataFrame(score_dict).sum(axis=1) + score_diff

    # Calculate holes remaining
    holes_remaining = constants.NUM_HOLES - hole["hole_number"] + 1

    # Store in a DataFrame to get hammer usage probability
    hammer_df = pd.DataFrame(
        {
            "score_diff": list(scores),
            "holes_remaining_prior": [holes_remaining]*len(scores),
            "hammers_used_prior": list(hammers)
        }
    )

    # Override hammer probability to 0 if they have 0 left
    hammer_df["hammer_probability"] = model_dict["model"].predict_proba(hammer_df[FEATURES])
    hammer_df["hammer_probability"] = [0 if hammer_count >= constants.MAX_HAMMERS else x for hammer_count, x in zip(hammer_df["hammers_used_prior"], hammer_df["hammer_probability"])]

    # Now for the other team (need to flip the score and pull hammers used for the right team)
    hammer_df["score_diff"] = -hammer_df["score_diff"]
    hammer_df["hammers_used_prior"] = list(other_hammers)
    hammer_df["other_hammer_probability"] = model_dict["model"].predict_proba(hammer_df[FEATURES])
    hammer_df["other_hammer_probability"] = [0 if hammer_count >= constants.MAX_HAMMERS else x for hammer_count, x in zip(hammer_df["hammers_used_prior"], hammer_df["other_hammer_probability"])]

    return hammer_df


def initialize_sim_probabilities(hole_df: pd.DataFrame) -> pd.DataFrame:
    """ This function sets the shooting and other team expected
    stroke probabilities at the start of a hole

    Args:
        hole_df (pd.DataFrame): DataFrame containing drive ex strokes
            and the stroke-level probabilities

    Returns:
        - hole_df (pd.DataFrame): DataFrame containing the above for the
            shooting and other team
    """

    rename_dict = {}
    
    # For simming holes, we simply use the off-the-tee expected stroke
    # probabilities. Since we don't factor in team/golfer strength, these
    # are the same for each team
    for col in hole_df.columns:
        if "_drive_stroke_prob" in col:
            split_col = col.split("_")[0]
            rename_dict[col] = "shooting_team_" + split_col + "_stroke_prob"
            hole_df["other_team_" + split_col + "_stroke_prob"] = list(hole_df[col])

    # Apply renaming derived above
    rename_dict["drive_ex_strokes"] = "shooting_team_ex_strokes"
    hole_df["other_team_ex_strokes"] = list(hole_df["drive_ex_strokes"])
    hole_df = hole_df.rename(columns=rename_dict)

    # Start-of-hole: nobody has hit yet and we're not on the green
    hole_df["shooting_team_strokes"] = 0
    hole_df["other_team_strokes"] = 0
    hole_df["shooting_team_one_putt_prob"] = np.nan
    hole_df["other_team_one_putt_prob"] = np.nan

    return hole_df


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
        current_hole (int): Hole that the teams are currently playing
        complete_hole (bool): Boolean whetehr to simulate a hole in progress or start from the tee
        sims (int): Number of match simulations to conduct

    Returns:
        tuple: A tuple containing the win and loss probabilities, along with the score differential
    """

    # Get current hole
    score_dict = {}
    base_outcomes = [-1, 0, 1]

    # Outcomes
    if complete_hole:
        # Sim rest-of-hole based on hole win/loss/tie probability
        outcomes = [outcome * hole_value for outcome in base_outcomes]
        probs = [shot["loss_probability"], shot["tie_probability"], shot["win_probability"]]

        # Sample outcomes and store
        shot_samples = np.random.choice(outcomes, size=sims, p=probs)
        score_dict[current_hole] = shot_samples

        # Rest of the holes
        hole_df = hole_df[hole_df["hole_number"] > current_hole]
    else:
        # Sim from start of the current hole (used to derive hammer value)
        hole_df = hole_df[hole_df["hole_number"] >= current_hole]

    # Get ex strokes and probabilities off the tee for the remaining holes
    if (current_hole < constants.NUM_HOLES) or (not complete_hole):

        # Pull ex strokes and rename columns accordingly
        hole_df = utils.get_drive_ex_strokes(hole_df).sort_values("hole_number")
        hole_df = initialize_sim_probabilities(hole_df)
        
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
            # These are hammers used across each simulation
            hammers = pd.DataFrame(hammer_dict).sum(axis=1)
            other_hammers = pd.DataFrame(other_hammer_dict).sum(axis=1)

            # For each simulation, the matches are in different states, so there's
            # a separate hammer deployment probabiilty for each
            hammer_prob_df = get_hammer_deployment_probabilities(
                hole, score_dict, score_diff, hammers, other_hammers, model_dict, sims
            )
            # For each match, sample one binomial trial for hammer deployment based on the
            # predicted probabilities
            hammer_samples = np.random.binomial(n=1, p=hammer_prob_df["hammer_probability"])
            other_hammer_samples = np.random.binomial(n=1, p=hammer_prob_df["other_hammer_probability"])

            # Initialize lists to store values of interest
            mask = []
            hammer_sample_list = []
            hammer_used_hole = []
            other_hammer_used_hole = []

            # For each simulated match, we need to track whether a hammer was thrown on each hole
            # and which team threw the hammer
            for h_sample, oh_sample, h_used, oh_used in zip(hammer_samples, other_hammer_samples, hammers, other_hammers):
                # Both teams threw the hammer, so we sample one binomial trail to determine
                # which team "threw it first" since our framework doesn't allow for three-point holes
                if (h_sample == 1 and h_used < constants.MAX_HAMMERS) and (oh_sample == 1 and oh_used < constants.MAX_HAMMERS):
                    mask.append(True)
                    # Randomize which team actually uses the hammer
                    binom_sample = np.random.binomial(n=1, p=0.5)
                    # Hammer thrown by the shooting team
                    if binom_sample == 1:
                        hammer_outcomes = list(value_df["realized_value"])
                        hammer_used_hole.append(1)
                        other_hammer_used_hole.append(0)
                    # Hammer thrown by the other team (need to flip the empirical
                    # realized point values)
                    else:
                        hammer_outcomes = list(-value_df["realized_value"])
                        hammer_used_hole.append(0)
                        other_hammer_used_hole.append(1)

                    # Sample the point value based on the empirical point value probabilities
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])
                # Shooting team throws the hammer
                elif h_sample == 1 and h_used < constants.MAX_HAMMERS:
                    mask.append(True)
                    hammer_outcomes = list(value_df["realized_value"])
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])

                    hammer_used_hole.append(1)
                    other_hammer_used_hole.append(0)
                # Other team throws the hammer
                elif oh_sample == 1 and oh_used < constants.MAX_HAMMERS:
                    mask.append(True)
                    hammer_outcomes = list(-value_df["realized_value"])
                    hammer_probs = list(value_df["probs"])
                    hammer_sample = np.random.choice(hammer_outcomes, size=1, p=hammer_probs)
                    hammer_sample_list.append(hammer_sample[0])
                    hammer_used_hole.append(0)
                    other_hammer_used_hole.append(1)
                # No hammers thrown
                else:
                    mask.append(False)
                    hammer_used_hole.append(0)
                    other_hammer_used_hole.append(0)

            # Get hole-level win/loss/tie probability to use when the hammer is not thrown
            win_prob, loss_prob, tie_prob = utils.calculate_win_loss_tie_probability(hole)

            # Sample hole outcomes
            probs = [win_prob, tie_prob, loss_prob]
            hole_samples = np.random.choice(base_outcomes, size=sims, p=probs)

            # Override the above sample hole outcomes with hammer outcomes when
            # the hammer is thrown
            hole_samples[mask] = hammer_sample_list
            score_dict[hole["hole_number"]] = hole_samples
            hammer_dict[hole["hole_number"]] = hammer_used_hole
            other_hammer_dict[hole["hole_number"]] = other_hammer_used_hole

    # Convert the dictionary storing all match scores to a DataFrame
    # and calculate the "total" score differential
    score_df = pd.DataFrame(score_dict)
    score_df["total"] = score_df.sum(axis=1) + score_diff

    # Split ties 50-50 since we don't simulate OT (different format)
    tie_prob = len(score_df[score_df["total"]==0])/len(score_df)
    win_prob = 0.5*tie_prob + len(score_df[score_df["total"]>0])/len(score_df)
    loss_prob = 0.5*tie_prob + len(score_df[score_df["total"]<0])/len(score_df)

    return win_prob, loss_prob, score_diff