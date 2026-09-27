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
from typing import List
import pickle

import pandas as pd

from pygam import s, GAM, te, LinearGAM, LogisticGAM
from pygam.distributions import BinomialDist


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

import sg_data
import viz_utils
import constants

# Model features (must be left in this order per the pygam setup)
FEATURES = ["score_diff", "holes_remaining_prior", "hammers_used_prior"]
TARGET = "hammer_used_on_hole"

def build_hammer_gam(df: pd.DataFrame, target_col: str = "hammer_used_on_hole", feature_cols: List = [], test_size: float = 0.2, random_state: int = 42):
    """
    Fit a LogisticGAM using the requested features, with a
    train/test split and cross-validated hyperparameter tuning, to predict
    the probability that a hammer will be deployed on a given hole

    Parameters:
        df (pd.DataFrame): DataFrame containing the modeling features and target.
        target_col (str): Column name for the target variable.
        feature_cols (list): List of model features
        test_size (float): Fraction of rows reserved for the test set.
        random_state (int): Random seed used for reproducibility.

    Returns:
        dict: A dictionary with the trained model and test
              split components.
    """

    # Break if we're missing required columns
    if not all(col in df.columns for col in feature_cols + [target_col]):
        missing = [col for col in feature_cols + [target_col] if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Design matrix
    X = df[feature_cols]
    y = df[target_col]

    # Split 'em
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit the model
    ## For the current feature set, this is a monotonically decreasing splint
    ## on score differential, then a tensor term for monotonically decreasing splines
    ## to interactthe number of hammers used and the number of holes remaining
    gam = LogisticGAM(s(0, n_splines=15, constraints="monotonic_dec") +
                     te(feature=(1, 2),
                        n_splines=(15, 15),
                        constraints=("monotonic_dec", "monotonic_dec")
                        ), lam=0.6).fit(X_train[feature_cols], y_train)

    # Hyperparameter tuning for lambda
    lams = [0.3, 0.6, 0.9]
    gam.gridsearch(X_train[feature_cols], y_train, lam=lams)

    return {
        "model": gam,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def construct_hammer_usage_gam():
    """main, ya cowboy"""

    # Pull in data
    hammer_df = sg_data.pull_hammers_remaining()

    # Isolate to fewer than three hammers used - if a team has
    # already used three hammers, they can't use anymore!
    hammer_df = hammer_df[hammer_df["hammers_used_prior"] < constants.MAX_HAMMERS]

    # Construct the model
    model_dict = build_hammer_gam(hammer_df, TARGET, FEATURES)

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
    viz_utils.visualize_calibration(test_df, TARGET)

    # Partial dependence
    viz_utils.plot_partial_dependence(model_dict["model"])

    # Break down tensor term
    holes_remaining = list(range(1, constants.NUM_HOLES + 1))
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
    viz_utils.visualize_hole_and_hammer_effects(temp_df)

    # Save model
    with open('models/hammer_model.pkl', 'wb') as handle:
        pickle.dump(model_dict, handle)


def build_hammer_opportunity_model():
    """ This function constructs a model to
    predict the probability of a hammer opportunity
    arising as a function of win probability. Changing the
    target variable below to future_hammer_ev from
    future_hammer_rate builds a model to predict the value
    of hammer opportunities in terms of win probability
    should they arise

    Returns:
        - prob_df (pd.DataFrame): DataFrame containing
            predicted probabilities of a hammer arising
            and the expected number of hammers given
            win probability and hammers remaining. If the
            target is future_hammer_ev, this predicts the average
            value of future hammer opportunities
    """

    # Define model features + target
    features = ["win_prob", "holes_remaining"]
    target = "future_hammer_ev"
    trials = "holes_remaining"

    # Pull data (no hammer opportunities after the last hole)
    hammer_df = sg_data.pull_future_hammer_value()
    hammer_df = hammer_df[hammer_df["hole_number"] < constants.NUM_HOLES]

    # Define proportions
    hammer_df["holes_remaining"] = constants.NUM_HOLES - hammer_df["hole_number"]
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
        binomial_gam.fit(train_df[features], train_df[target])
        train_preds = binomial_gam.predict(train_df[features])
        test_preds = binomial_gam.predict(test_df[features])
    else:
        gam = LinearGAM(te(0, 1), constraints=("concave", None))
        gam.fit(train_df[features], train_df[target])
        train_preds = gam.predict(train_df[features])
        test_preds = gam.predict(test_df[features])

    # Derive counts if we're predicting hammer opportunities
    if target == "future_hammer_rate":
        train_counts = train_preds * train_df[trials]
        test_counts = test_preds * test_df[trials]


    # Performance
    print("Training MAE: " + str(round(mean_absolute_error(train_df[target],
                                                           train_preds), 3)))
    print("Test MAE: " + str(round(mean_absolute_error(test_df[target],
                                                        test_preds), 3)))

    if target == "future_hammer_rate":
        # Count performance
        print("Training Counts MAE: " + str(round(mean_absolute_error(train_df[target] * train_df[trials],
                                                                train_counts), 3)))
        print("Test Counts MAE: " + str(round(mean_absolute_error(test_df[target] * test_df[trials],
                                                            test_counts), 3)))

        # Plot partial dependence
        viz_utils.plot_partial_dependence(binomial_gam)

        # Plot calibration
        test_df["preds"] = test_counts
        viz_utils.visualize_count_calibration(test_df, "future_hammer_opportunities", "preds")

        # Save model
        with open('models/hammer_opps_model.pkl', 'wb') as handle:
            pickle.dump(binomial_gam, handle)
    else:
        # Plot partial dependence
        viz_utils.plot_partial_dependence(gam)

        # Plot calibration
        train_df["preds"] = train_preds
        viz_utils.visualize_count_calibration(train_df, target, "preds")

        # Save model
        with open('models/hammer_ev_model.pkl', 'wb') as handle:
            pickle.dump(gam, handle)
