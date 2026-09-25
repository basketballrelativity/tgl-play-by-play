"""
This file contains functionaility to produce
visualizations for the TGL models
"""
import os

import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Plotting info
LOGO_DIR = "images"

# Same color family:
#   Observed = lighter
#   Optimal  = darker
OBSERVED_COLOR = "#8BAFD1"
OPTIMAL_COLOR = "#245A9C"

# Colors
TEXT_COLOR = "#222222"
SECONDARY_TEXT = "#666666"
GRID_COLOR = "#D9D9D9"
SEPARATOR_COLOR = "#E8E8E8"
ZERO_LINE_COLOR = "#444444"

# Font and size
FONT_FAMILY = "DejaVu Sans"

TITLE_SIZE = 20
SUBTITLE_SIZE = 11
LEGEND_SIZE = 9
AXIS_SIZE = 10

# Fig params
FIG_WIDTH = 11
FIG_HEIGHT = 8

BAR_HEIGHT = 0.28
BAR_GAP = 0.10


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


def visualize_count_calibration(data_df, target_label, pred_label):
    """
    This function visualizes calibration for a continuous variable prediction

    @param data_df (DataFrame): DataFrame containing
        columns for target_label and pred_label
    @param target_label (str): Name of the column holding the target variable
    @param pred_label (str): Name of the column holding the predicted variable

    Returns:

        fig (plt.figure): Figure object of the calibration
            visualization
    """

    # Initialize discretizer
    overall_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', subsample=None)

    # Fit and transform to bins
    pred_bin = overall_est.fit_transform(data_df[[pred_label]])

    # Store
    data_df["bin"] = [x[0] for x in pred_bin]
    overall_viz = pd.DataFrame(data_df.groupby("bin")[[target_label, pred_label]].mean()).reset_index()

    fig, ax = plt.subplots()

    # Plot
    ax.plot(overall_viz[pred_label], overall_viz[target_label], linestyle='-', marker='o', markersize=3, color='gray')

    # Add a diagonal line for perfect calibration
    min_val = min(overall_viz[pred_label].min(), overall_viz[target_label].min())
    max_val = max(overall_viz[pred_label].max(), overall_viz[target_label].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label="Perfect Calibration")

    # Labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Calibration Plot')
    ax.legend()

    plt.savefig(f"{target_label}_value_calibration.png")
    plt.close(fig)


def get_logo_labels(ax, values, positions):
    """
    This function is used to position the TGL team logos
    to the right of the horizontal bars

    Args:
        ax (matplotlib.axes): Axis object for the figure
        values (list): List of x-values corresponding to the
            horizontal position of the bar
        positions (list): List of y-values corresponding to
            the vertical position of the bar

    Returns:
        - tuple of (x, y) coordinates for where the logo should
            be placed
    """

    # Determine an appropriate offset based on the chart range.
    xmin, xmax = ax.get_xlim()
    offset = (xmax - xmin) * 0.008

    x_val = []
    y_val = []

    # Loop through bars and derive the appropriate
    # coordinates for the logos
    for value, y_pos in zip(values, positions):

        if value >= 0:
            x = value + offset + 0.05
        else:
            x = value - offset + 0.05

        x_val.append(x)
        y_val.append(y_pos)

    return x_val, y_val
        


def add_team_logo(ax, logo_path, x, y):
    """
    Add a team logo at (x, y), with all logos normalized to the
    same maximum physical dimension.

    Args:
        ax (matplotlib.axes): Axis object for the figure
        logo_path (str): Local path to the logo file
        x (float): x-coordinate where the logo should be placed
        y (float): y-coordinate where the logo should be placed
        zoom

    Returns:
        None, but updates the figure object accordingly
    """

    # Open the image
    img = Image.open(logo_path).convert("RGBA")

    # High-quality initial resize via Pillow (Keep this)
    max_dimension = 500
    width, height = img.size
    scale = max_dimension / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize logo
    img = img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # 2. CHANGE: Switch interpolation to "nearest" to prevent double-smoothing
    imagebox = OffsetImage(
        img,
        interpolation="nearest", 
    )

    # Plot logo
    annotation = AnnotationBbox(
        imagebox,
        (x, y),
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
    )

    ax.add_artist(annotation)



def viz_wpa(data_df):
    """ This function visualizes the win probability added
    for observed and optimal hammer usage

    Args:
        data_df (pd.DataFrame): DataFrame containing optimal and
            observed win probability added

    Returns
        None, but saves `wpa.png` to the local directory
    """

    # Sort DataFrame and extract lists for fields of interest
    data_df = data_df.sort_values(
        "optimal_wpa",
        ascending=True
    ).reset_index(drop=True)

    teams = data_df["teamId"].tolist()
    observed = data_df["observed_wpa"].tolist()
    optimal = data_df["optimal_wpa"].tolist()

    n_teams = len(data_df)

    # Set figure settings
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": AXIS_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


    # Initialize figure
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )


    # Vertical locations
    # Middle of the bars
    y = np.arange(n_teams)

    # Observed bar
    observed_y = y - (
        BAR_HEIGHT / 2 + BAR_GAP / 2
    )

    # Optimal bar
    optimal_y = y + (
        BAR_HEIGHT / 2 + BAR_GAP / 2
    )


    # Plot bars
    ax.barh(
        observed_y,
        observed,
        height=BAR_HEIGHT,
        color=OBSERVED_COLOR,
        edgecolor="none",
        label="Observed usage",
        zorder=3,
    )

    ax.barh(
        optimal_y,
        optimal,
        height=BAR_HEIGHT,
        color=OPTIMAL_COLOR,
        edgecolor="none",
        label="Optimal usage",
        zorder=3,
    )


    # Calibrate horizontal range
    all_values = np.concatenate([
        observed,
        optimal,
    ])

    data_min = all_values.min()
    data_max = all_values.max()

    data_range = data_max - data_min

    if data_range == 0:
        data_range = 0.01

    # Padding around the actual data
    left_padding = data_range * 0.12
    right_padding = data_range * 0.08

    xmin = min(0, data_min) - left_padding
    xmax = max(0, data_max) + right_padding

    ax.set_xlim(xmin, xmax)

    # Plot zero line
    ax.axvline(
        0,
        color=ZERO_LINE_COLOR,
        linewidth=1.1,
        zorder=2,
    )

    # Separate teams
    for y_pos in np.arange(n_teams - 1) + 0.5:

        ax.axhline(
            y_pos,
            color=SEPARATOR_COLOR,
            linewidth=0.7,
            zorder=1,
        )

    # Construct grid
    ax.grid(
        axis="x",
        color=GRID_COLOR,
        linewidth=0.7,
        linestyle="-",
        zorder=0,
    )

    ax.set_axisbelow(True)


    # No team labels.
    ax.set_yticks(y)
    ax.set_yticklabels([])

    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
    )

    # X-axis
    ax.tick_params(
        axis="x",
        which="major",
        length=0,
        pad=8,
        colors=SECONDARY_TEXT,
        labelsize=AXIS_SIZE,
    )

    # WPA displayed as percentage points
    ax.xaxis.set_major_formatter(
        lambda x, pos: f"{x:.0%}"
    )

    ax.set_xlabel(
        "Win Probability Added",
        fontsize=AXIS_SIZE,
        color=SECONDARY_TEXT,
        labelpad=10,
    )


    # Set axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color("#BDBDBD")
    ax.spines["bottom"].set_linewidth(0.8)


    # Position logos
    logo_x, logo_y = get_logo_labels(
                ax,
                optimal,
                y,
            )

    for i, team in enumerate(teams):
        clean_team = team.replace("tgl", "")
        logo_path = os.path.join(
            LOGO_DIR,
            f"{clean_team}.avif",
        )

        if not os.path.exists(logo_path):
            print(
                f"Warning: logo not found for {team}: "
                f"{logo_path}"
            )
            continue

        add_team_logo(
            ax,
            logo_path,
            logo_x[i],
            logo_y[i],
        )


    # Write title
    fig.text(
        0.075,
        0.965,
        "Hammer Usage Win Probability Added",
        ha="left",
        va="top",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
    )


    # Write subtitle
    fig.text(
        0.075,
        0.925,
        "Observed versus optimal hammer usage by team in the 2026 TGL season",
        ha="left",
        va="top",
        fontsize=SUBTITLE_SIZE,
        color=SECONDARY_TEXT,
    )


    # Plot legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.075),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.3,
        handleheight=0.8,
        columnspacing=1.5,
        borderaxespad=0,
    )


    # Tighten layout
    plt.subplots_adjust(
        left=0.075,
        right=0.965,
        top=0.84,
        bottom=0.10,
    )

    # Save and close
    plt.savefig(
        f"wpa.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
        )
    plt.close(fig)