import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import numpy as np
import pandas as pd

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

def plot_monthly_evolution(df, date_col, value_col,
                           title="Monthly Trend",
                           xlabel="Date",
                           ylabel="Value"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        x=date_col,
        y=value_col,
        data=df,
        marker="o",
        color="steelblue",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    return fig


# def plot_bar_features(df, features,
#                       title="Bar Plots of Categorical, Discrete and Date Features",
#                       n_cols=3):
#     sns.set(style="white")

#     # Calculate number of rows automatically
#     n_rows = (len(features) + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
#     axes = axes.flatten()

#     for ax, feat in zip(axes, features):
#         sns.countplot(x=df[feat],
#                       order=df[feat].value_counts().index,
#                       color="#66a4de",
#                       edgecolor="black",
#                       ax=ax)
#         ax.set_title(feat)
#         ax.set_xlabel('')
#         ax.tick_params(axis='x', rotation=45)

#     # Delete extra axes if features < subplot spaces
#     for j in range(len(features), len(axes)):
#         fig.delaxes(axes[j])

#     fig.suptitle(title, fontsize=25, fontweight="bold")
#     fig.tight_layout(rect=[0, 0, 1, 0.97])

#     return fig

