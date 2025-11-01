import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import numpy as np
import pandas as pd

def boxplots_CDB(df: pd.DataFrame, features: list, rows: int = 2, figsize=(20, 10)):
    """
    Create a matplotlib Figure containing boxplots for the given features from df.
    - df: customer dataframe
    - features: list of numeric feature names to plot
    - rows: preferred number of rows (function will compute columns automatically)
    Returns: matplotlib.figure.Figure
    """
    if not features:
        raise ValueError("No features provided for boxplots.")

    n = len(features)
    cols = ceil(n / rows)

    sns.set_style("white")
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.2)

    # Flatten axes to iterate easily
    if hasattr(axes, "flatten"):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    # Hide extra axes if any
    for ax in axes_list[n:]:
        ax.set_visible(False)

    for ax, feat in zip(axes_list, features):
        if feat not in df.columns:
            ax.set_visible(False)
            continue

        data = df[feat].dropna().reset_index(drop=True)
        if data.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            x=data,
            ax=ax,
            color="#0062FF",
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=3),
            flierprops=dict(markeredgecolor="black")
        )

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(feat, fontsize=18)
        ax.set_xlabel("")

    plt.suptitle("Metric Variables' Box Plots", fontsize=25, fontweight="bold")
    return fig


def location_scatter_CDB(df: pd.DataFrame, lon_col: str = "Longitude", lat_col: str = "Latitude",
                         show_trend: bool = True, figsize=(10, 6)):
    """
    Scatter plot of Longitude vs Latitude using ALL available points.
    - df: dataframe
    - lon_col, lat_col: column names for longitude & latitude
    - show_trend: whether to fit and show a linear trend line
    Returns: matplotlib.figure.Figure
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        raise KeyError(f"Missing coordinates: expected '{lon_col}' and '{lat_col}' in dataframe.")

    coords = df[[lon_col, lat_col]].dropna()
    if coords.empty:
        raise ValueError("No coordinate data available for scatter plot.")

    x = coords[lon_col].values
    y = coords[lat_col].values

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=25, alpha=0.3, c='blue', edgecolor='k', linewidth=0.3)

    # Trend line if requested and we have variation
    try:
        if show_trend and len(x) >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)
            sort_idx = np.argsort(x)
            ax.plot(x[sort_idx], trend(x[sort_idx]), color='red', linewidth=2)
    except Exception:
        pass

    ax.set_title("Customer Locations", fontsize=18, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=14, labelpad=12)
    ax.set_ylabel("Latitude", fontsize=14, labelpad=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
