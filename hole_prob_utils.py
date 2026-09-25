"""
This script contains the functionality to
predict whether a hole will be won, lost,
or tied
"""

import pandas as pd
import numpy as np

# Maximum number of strokes/putts to hole out
MAX_STROKES = 10
MAX_PUTTS = 3

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

    # If expected strokes is zero, we don't need probabilities
    if shot[f"{team}_team_ex_strokes"] == 0:
        team_probs = [np.nan] * MAX_STROKES  # Already holed out
    elif pd.notnull(shot[f"{team}_team_one_putt_prob"]):
        # If the putting probs aren't null, we're puttin'
        two_putt_prob = 1 - shot[f"{team}_team_one_putt_prob"] - shot[f"{team}_team_three_putt_prob"]
        team_probs = [
            shot[f"{team}_team_one_putt_prob"],
            two_putt_prob,
            shot[f"{team}_team_three_putt_prob"],
        ] + [0] * (MAX_STROKES - MAX_PUTTS)  # Fill the rest with zeros
    else:
        # Otherwise, pull the full stroke probability vector
        team_probs = []
        for i in range(1, MAX_STROKES + 1):
            prob = shot.get(f"{team}_team_{i}_stroke_prob")
            team_probs.append(prob if pd.notnull(prob) else 0)
    
    return team_probs


def finish_hole_one_team(team_probs, team_strokes, opponent_strokes):
    """ This function finishes the hole if one team has already holed out

    Args:
        shot (pd.Series): Series object containing the shot information
        team_probs (list): Vector of expected stroke probabilities for the team yet
            to hole out
        team_strokes (int): Number of strokes taken by the team yet to hole out
        opponent_strokes (int): Number of strokes taken by the team that has holed
            out

    Returns:
        - Tuple of win, loss, and tie probabilities corresponding to the team
            yet to hole out
    """

    # Initialize probabilities
    win_prob = 0.0
    loss_prob = 0.0
    tie_prob = 0.0

    for putts, prob in zip(range(1, MAX_STROKES + 1), team_probs):
        if prob <= 0:
            continue
        # Team total
        total = team_strokes + putts

        # Win
        if total < opponent_team_strokes:
            win_prob += prob
        # Lose
        elif total > opponent_team_strokes:
            loss_prob += prob
        # Draw
        else:
            tie_prob += prob

    total_prob = win_prob + loss_prob + tie_prob
    # Normalize
    if total_prob > 0:
        win_prob /= total_prob
        loss_prob /= total_prob
        tie_prob /= total_prob

    return win_prob, loss_prob, tie_prob


def calculate_win_loss_tie_probability(shot: pd.Series):
    """
    Calculate the probability that one team finishes with fewer total strokes than
    the other.

    Args:
        shot (pd.Series): A pandas Series containing the shot-level data for a specific shot.
    
    Returns:
        tuple: A tuple containing the win, loss, and tie probabilities.
    """

    # Pull the stroke probability vectors for each team
    shooting_team_probs = get_probability_vectors(shot, "shooting")
    other_team_probs = get_probability_vectors(shot, "other")

    # Get actual strokes up to this shot
    shooting_team_strokes = shot["shooting_team_strokes"]
    other_team_strokes = shot["other_team_strokes"]

    # The hole is over in this case, so sort out what happened
    if shot["shooting_team_ex_strokes"] == 0 and shot["other_team_ex_strokes"] == 0:
        if shooting_team_strokes < other_team_strokes:
            return 1.0, 0.0, 0.0
        elif shooting_team_strokes > other_team_strokes:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 1.0

    # The shooting team has already holed out
    if shot["shooting_team_ex_strokes"] == 0:
        loss_prob, tie_prob, win_prob = finish_hole_one_team(
            other_team_probs, other_team_strokes, shooting_team_strokes
        )
        # Note that these are returned with win and loss probs swapped because
        # we want to return win, loss, and tie probability relative to the shooting
        # team
        return win_prob, loss_prob, tie_prob

    # The other team has already holed out
    if shot["other_team_ex_strokes"] == 0:
       win_prob, tie_prob, loss_prob = finish_hole_one_team(
            shooting_team_probs, shooting_team_strokes, other_team_strokes
        )
        return win_prob, loss_prob, tie_prob

    # Otherwise, both teams are still playing
    win_prob = 0.0
    loss_prob = 0.0
    tie_prob = 0.0

    for shooting_putts, shooting_prob in  zip(range(1, MAX_STROKES + 1), shooting_team_probs):
        for other_putts, other_prob in  zip(range(1, MAX_STROKES + 1), other_team_probs):
            if shooting_prob <= 0 and other_prob <= 0:
                continue

            # Get each team's total
            shooting_total = shooting_team_strokes + shooting_putts
            other_total = other_team_strokes + other_putts

            # Independence assumption to get the probability of each
            # combination of scores
            if shooting_total < other_total:
                win_prob += shooting_prob * other_prob
            elif shooting_total > other_total:
                loss_prob += shooting_prob * other_prob
            else:
                tie_prob += shooting_prob * other_prob

    # Normalize
    total_prob = win_prob + loss_prob + tie_prob
    if total_prob > 0:
        win_prob /= total_prob
        loss_prob /= total_prob
        tie_prob /= total_prob

    return win_prob, loss_prob, tie_prob