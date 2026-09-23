"""
This script contains the functions to
build, test, and deploy a hammer deployment
probability model on TGL data

# main function that
## Pulls in hammer data from sg_data
## Constructs the model
## Evaluates performance on train + test set (log loss and calibration)
## Examines partial dependence plots and hypothetical predictions
## Saves model
"""
import pickle

import pandas as pd
import numpy as np

from pygam import s, GAM, te, LinearGAM
from pygam.distributions import BinomialDist
from statsmodels.miscmodels.ordinal_model import OrderedModel


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

import sg_data
import utils

MAX_HAMMERS = 3 # Maximum number of hammers available to use in a match
# Model features (must be left in this order per the pygam setup)
FEATURES = ["score_diff", "holes_remaining_prior", "hammers_used_prior"]
TARGET = "hammer_used_on_hole"

def plot_partial_dependence(gam):
    """
    """

    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue

        if "te" in repr(term):
            meshgrid = True
            XX = gam.generate_X_grid(term=i, meshgrid=meshgrid)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95, meshgrid=meshgrid)
            ax = plt.axes(projection="3d")
            ax.plot_surface(XX[0], XX[1], pdep, cmap="viridis")
        else:
            meshgrid = False
            XX = gam.generate_X_grid(term=i, meshgrid=meshgrid)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95, meshgrid=meshgrid)

            plt.figure()
            plt.plot(XX[:, term.feature], pdep)
            plt.plot(XX[:, term.feature], confi, c="r", ls="--")
        
        plt.title(repr(term))
        plt.savefig(f"{repr(term)}_partial_dependence.png")
        plt.close()


def visualize_hole_and_hammer_effects(pred_df: pd.DataFrame):
    """ This function visualizes model output as a function
    of holes and hammers remaining

    Args:
        pred_df (pd.DataFrame): DataFrame containing model features
            and predicted output

    Returns:
        matplotlib chart saved locally
    """

    # Subset data
    zero_hammers_used = pred_df[
        pred_df["hammers_used_prior"]==0
    ]
    one_hammers_used = pred_df[
        pred_df["hammers_used_prior"]==1
    ]
    two_hammers_used = pred_df[
        pred_df["hammers_used_prior"]==2
    ]

    # Plot each trend line
    fig = plt.figure()
    plt.plot(
        zero_hammers_used["holes_remaining_prior"],
        zero_hammers_used["preds"], label="0 Hammers Used", color='tab:blue', linestyle="-"
    )
    plt.plot(
        one_hammers_used["holes_remaining_prior"],
        one_hammers_used["preds"], label="1 Hammer Used", color='tab:blue', linestyle="--"
    )
    plt.plot(
        two_hammers_used["holes_remaining_prior"],
        two_hammers_used["preds"], label="2 Hammers Used", color='tab:blue', linestyle="-."
    )

    # Title and axes
    plt.title("Hammer Use Probability by Holes and Hammers Reamining")
    plt.xlabel("Holes Remaining")
    plt.ylabel("Hammer Use Probability (tie score)")

    # Legend
    plt.legend()

    # Wrap it up
    plt.savefig("hammer_use_probabiliy.png")
    plt.close()


def main():
    """main, ya cowboy"""

    # Pull in data
    hammer_df = sg_data.pull_hammers_remaining()

    # Isolate to fewer than three hammers used - if a team has
    # already used three hammers, they can't use anymore!
    hammer_df = hammer_df[hammer_df["hammers_used_prior"] < MAX_HAMMERS]

    # Construct the model
    model_dict = utils.build_hammer_gam(hammer_df, TARGET, FEATURES)

    # Evaluate performance
    train_preds = model_dict["model"].predict_proba(model_dict["X_train"])
    test_preds = model_dict["model"].predict_proba(model_dict["X_test"])

    # Log loss
    print("Training Set Log Loss: " + str(round(log_loss(model_dict["y_train"],
                                                           train_preds), 3)))
    print("Test Set Log Loss: " + str(round(log_loss(model_dict["y_test"],
                                                        test_preds), 3)))
    
    # Calibration
    test_df = pd.DataFrame(model_dict["X_test"], columns=FEATURES)
    test_df[TARGET] = list(model_dict["y_test"])
    test_df[TARGET + "_probability"] = test_preds
    utils.visualize_calibration(test_df, TARGET)

    # Partial dependence
    plot_partial_dependence(model_dict["model"])

    # Break down tensor term
    holes_remaining = list(range(1, 16))
    temp_df = pd.DataFrame(
        {
            "holes_remaining_prior": holes_remaining * 3,
            "hammers_used_prior": [0]*len(holes_remaining) +
                [1]*len(holes_remaining) +
                [2]*len(holes_remaining)
        }
    )
    temp_df["score_diff"] = 0
    temp_df["preds"] = model_dict["model"].predict_proba(temp_df[FEATURES])

    # Visualize predictions as a function of holes and hammers remaining
    visualize_hole_and_hammer_effects(temp_df)

    # Save model
    with open('hammer_model.pkl', 'wb') as handle:
        pickle.dump(model_dict, handle)


def build_hammer_opportunity_model():
    """ This function constructs a model to
    predict the probability of a hammer opportunity
    arising as a function of win probability

    Returns:
        - prob_df (pd.DataFrame): DataFrame containing
            predicted probabilities of a hammer arising
            and the expected number of hammers given
            win probability and hammers remaining
    """

    # Define model features + target
    features = ["win_prob", "holes_remaining"]
    target = "future_hammer_rate"
    trials = "holes_remaining"

    # Pull data (no hammer opportunities after the last hole)
    hammer_df = sg_data.pull_future_hammer_value()
    hammer_df = hammer_df[hammer_df["hole_number"] < 15]

    # Define proportions
    hammer_df["holes_remaining"] = 15 - hammer_df["hole_number"]
    hammer_df["future_hammer_rate"] = (
        hammer_df["future_hammer_opportunities"] /
        hammer_df["holes_remaining"]
    )

    # Split data
    train_df, test_df = train_test_split(hammer_df, test_size=0.2, random_state=42)

    # Define model
    if target == "future_hammer_rate":
        binomial_gam = GAM(te(0, 1, constraints=("concave", "monotonic_dec")), distribution=BinomialDist(), link='logit')

        # Fit model
        binomial_gam.fit(train_df[features], train_df[target])#, weights=train_df[trials])
    else:
        gam = LinearGAM(te(0, 1, constraints=("concave", "monotonic_dec")))
        gam.fit(train_df[features], train_df[target])

    # Evaluate
    train_preds = binomial_gam.predict(train_df[features])
    test_preds = binomial_gam.predict(test_df[features])

    if target == "future_hammer_rate":
        train_counts = train_preds * train_df[trials]
        test_counts = test_preds * test_df[trials]


    print("Training MAE: " + str(round(mean_absolute_error(train_df[target],
                                                           train_preds), 3)))
    print("Test MAE: " + str(round(mean_absolute_error(test_df[target],
                                                        test_preds), 3)))

    if target == "future_hammer_rate":
        print("Training Counts MAE: " + str(round(mean_absolute_error(train_df[target] * train_df[trials],
                                                                train_counts), 3)))
        print("Test Counts MAE: " + str(round(mean_absolute_error(test_df[target] * test_df[trials],
                                                            test_counts), 3)))

        # Plot partial dependence
        plot_partial_dependence(binomial_gam)

        # Plot calibration
        test_df["preds"] = test_counts
        utils.visualize_count_calibration(test_df, "future_hammer_opportunities", "preds")

        # Save model
        with open('hammer_opps_model.pkl', 'wb') as handle:
            pickle.dump(binomial_gam, handle)
    else:
        # Plot partial dependence
        plot_partial_dependence(gam)

        # Plot calibration
        test_df["preds"] = test_preds
        utils.visualize_count_calibration(test_df, target, "preds")

        # Save model
        with open('hammer_ev_model.pkl', 'wb') as handle:
            pickle.dump(gam, handle)


# if __name__ == "__main__":
#     main()