import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np






# --------------------------------------- PROFILING GRAPHS --------------------------------------#

def plot_profiling_barplots(var_name, ohe_profile, groups, title_suffix='Merged Segment'):
    """ Plot stacked barplots for a specific group of one-hot encoded features.
    Parameters:
        var_name : Base feature name to plot (e.g. 'City', 'Province or State').
        ohe_profile : Cluster-level profiling dataframe (rows = clusters, columns = OHE features).
        groups : Dictionary mapping base feature names to lists of OHE columns.
        title_suffix : Suffix to add to the plot title (default is 'Merged Segment').
    Returns:
        Displays a stacked bar plot for the specified variable.
    """
    # Get the columns for the specified variable
    cols = groups[var_name]

    # # Ensure integer cluster labels
    # ohe_profile = ohe_profile.copy()
    # ohe_profile.index = ohe_profile.index.astype(int)

    # Plot stacked barplot
    ohe_profile[cols].plot(
        kind='bar',
        stacked=True,
        ax=plt.gca(),
        color=sns.color_palette("Set2", n_colors=len(cols))
    )

    # Add title and labels
    plt.title(f'{var_name} Distribution by {title_suffix}', fontsize=12)
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')

    # Add legend
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        frameon=False
    )

def plot_profiling_heatmaps(ohe_profile, groups, group_keys, figsize=(12, 8), cmap="mako"):
    """ Plot heatmaps for one or more groups of one-hot encoded categorical features.
    Parameters:
        ohe_profile : Cluster-level profiling dataframe (rows = clusters, columns = OHE features).
        groups : Dictionary mapping feature group names to lists of corresponding one-hot encoded columns.
        group_keys : List of categorical feature groups to plot (e.g. ['Income_Bins', 'Marital Status']).
        figsize : Size of the entire figure (default is (12, 8)).
        cmap : Colormap to use for the heatmaps (default is "mako").
    Returns:
        Displays heatmaps for the specified groups.
    """

    # Ensure list input
    if not isinstance(group_keys, list):
        raise ValueError("group_keys must be a list of strings.")

    # Ensure integer cluster labels
    ohe_profile = ohe_profile.copy()
    ohe_profile.index = ohe_profile.index.astype(int)

    # number of plots
    n_plots = len(group_keys)

    # create subplots
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=figsize,
        sharey=True
    )

    # set overall title
    fig.suptitle(
    "Categorical Profiling by Cluster - Heatmaps",
    fontsize=16,
    fontweight='bold',
    y=1.02)

    # adjust axes for single plot case
    if n_plots == 1:
        axes = [axes]

    # plot each heatmap
    for ax, key in zip(axes, group_keys):
        if key not in groups:
            raise ValueError(f"'{key}' not found in groups dictionary.")

        cols = groups[key]

        sns.heatmap(
            ohe_profile[cols],
            ax=ax,
            cmap=cmap,
            linewidths=0.5,
            vmin=0,          # adjust to minimum real of your data
            vmax=0.40,       # adjust to maximum real of your data
            cbar=ax is axes[-1],
            cbar_kws={"label": "Proportion"}
        )

        ax.set_title(f"{key} Distribution by Cluster", fontsize=12, fontweight="bold")
        ax.set_xlabel(key)

    # set shared y-label
    axes[0].set_ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_profiling_radar(group_keys, profile_df, groups, alpha=0.2):
    """ Plot radar charts for one or more groups of one-hot encoded categorical features, sharing a single legend.
    Parameters
        group_keys : Categorical feature groups to plot (e.g. ['Income_Bins', 'Marital Status']).
        profile_df : Cluster-level profiling dataframe (rows = clusters, columns = OHE features).
        groups : Dictionary mapping feature group names to lists of corresponding one-hot encoded columns.
        alpha : Transparency level for the filled radar areas.
    Returns:
        Displays radar charts for the specified groups.
    """

    # Ensure list
    if isinstance(group_keys, str):
        group_keys = [group_keys]

    # # Ensure integer cluster labels
    # profile_df = profile_df.copy()
    # profile_df.index = profile_df.index.astype(int)

    # number of plots
    n_plots = len(group_keys)

    # create subplots
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(6 * n_plots, 6),
        subplot_kw=dict(polar=True)
    )

    # set overall title
    fig.suptitle(
    "Categorical Profiling by Cluster - Radar Charts",
    fontsize=16,
    fontweight='bold',
    y=1.02)
    
    # adjust axes for single plot case
    if n_plots == 1:
        axes = [axes]

    # plot each graph
    for ax, group_key in zip(axes, group_keys):
        if group_key not in groups:
            raise ValueError(f"Group '{group_key}' not found in groups.")

        features = groups[group_key]
        n_features = len(features)

        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        angles = np.append(angles, angles[0])

        for cluster_id in profile_df.index:
            values = profile_df.loc[cluster_id, features].values
            values = np.append(values, values[0])

            ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_id}')
            ax.fill(angles, values, alpha=alpha)

        ax.set_thetagrids(angles[:-1] * 180 / np.pi, features)
        ax.set_title(group_key, y=0.96, fontsize=10, fontweight="bold")
        ax.grid(True)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(profile_df.index),
        frameon=False,
        fontsize=10
    )

    fig.subplots_adjust(
        top=0.80,
        bottom=0.10,
        wspace=0.35
    )

    plt.show()

