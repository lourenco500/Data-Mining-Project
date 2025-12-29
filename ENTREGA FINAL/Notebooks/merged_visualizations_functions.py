import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.cm import get_cmap
import matplotlib as mpl
import numpy as np


# --------------------------------------- PCA GRAPHS --------------------------------------#

def plot_pca_heatmaps(loadings, pc_corr, figsize=(16, 9), cmap="coolwarm"):
    """Plot side-by-side heatmaps for PCA loadings and feature-component correlations.
    Parameters:
        loadings : PCA component loadings (features x components).
        pc_corr : Feature-component correlation matrix.
        figsize : Figure size.
        cmap : Diverging colormap.
    Returns:
        Displays the heatmaps.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # PCA Loadings
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Loading"},
        ax=axes[0]
    )
    axes[0].set_title("PCA Component Loadings", fontsize=16, weight="bold", pad=15)
    axes[0].set_xlabel("Principal Components")

    # PCA Correlations
    sns.heatmap(
        pc_corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=axes[1]
    )
    axes[1].set_title("Feature–Component Correlation", fontsize=16, weight="bold", pad=15)
    axes[1].set_xlabel("Principal Components")

    plt.tight_layout()
    plt.show()


def plot_pca_2d_comparison(pca_with_income, pca_without_income, labels, figsize=(18, 10), colormap="Set2"):
    """ Plot side-by-side 2D PCA projections with and without Income feature, including cluster centroids.
    Parameters:
        pca_with_income : 2D PCA data including Income feature.
        pca_without_income : 2D PCA data excluding Income feature.
        labels : Cluster labels for coloring points.
        figsize : Figure size.
        colormap : Colormap for clusters.
    Returns:
        Displays the PCA scatter plots.
    """
    # create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    clusters = sorted(pd.unique(labels))

    # plot each PCA projection
    for ax, pca_data, title, show_cbar in zip(
        axes,
        [pca_with_income, pca_without_income],
        ["2D PCA With Income Feature", "2D PCA Without Income Feature"],
        [False, True]
    ):
        pd.DataFrame(pca_data).plot.scatter(
            x=0,
            y=1,
            c=labels,
            ax=ax,
            colormap=colormap,
            colorbar=show_cbar
        )

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
        ax.grid(False)

        # Centroids
        for c in clusters:
            mask = labels == c
            cx = pca_data[mask, 0].mean()
            cy = pca_data[mask, 1].mean()

            ax.scatter(cx, cy, c="black", marker="X", s=200, zorder=10)
            ax.annotate(
                f"C{c}",
                (cx, cy),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold"
            )

    plt.tight_layout()
    plt.show()


# --------------------------------------- t-SNE GRAPHS --------------------------------------#

def plot_tsne_2d(tsne_embedding, labels, title, width=900, height=650, palette=px.colors.qualitative.Set2):
    """Function to plot 2D t-SNE embeddings with Plotly scatter plot.
    Parameters:
        tsne_embedding: 2D numpy array of t-SNE coordinates.
        labels: Pandas Series or array-like of cluster labels for coloring points.
        title: Title of the plot.
        width: Width of the plot in pixels.
        height: Height of the plot in pixels.
        palette: List of colors for clusters.
    Returns:
        Plotly Figure object.
    """

    # Build plotting DataFrame and explicitly align t-SNE points with cluster labels
    # Using the same index avoids misalignment issues that can lead to NaN clusters
    df_tsne = pd.DataFrame(
        tsne_embedding,
        columns=["t-SNE 1", "t-SNE 2"],
        index=labels.index
    )

    # Convert cluster labels to strings so Plotly treats them as categorical variables
    df_tsne["cluster"] = labels.astype(str)

    # Create 2D scatter plot: points are colored by cluster membership
    fig = px.scatter(
        df_tsne,
        x="t-SNE 1",
        y="t-SNE 2",
        color="cluster",
        title=title,
        color_discrete_sequence=palette,
        # Ensure clusters appear in a consistent order in the legend
        category_orders={
            "cluster": sorted(df_tsne["cluster"].unique(), key=str)
        },
        width=width,
        height=height
    )

    # Marker size and transparency adjusted to reduce overplotting
    fig.update_traces(marker=dict(size=6, opacity=0.65))

    # Global layout styling to match an academic / publication-ready look
    fig.update_layout(
        title_x=0.5,
        font=dict(size=14, family="Arial"),
        legend_title_text="Cluster",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Axis styling: remove grid, add clear axis lines
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    )

    # Enforce equal scaling on both axes so geometric relationships are preserved
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        scaleanchor="x",
        scaleratio=1
    )

    fig.show()


# --------------------------------------- UMAP GRAPHS --------------------------------------#

def plot_umap(
    umap_embedding,
    labels,
    title,
    dims=2,
    width=900,
    height=600,
    palette=px.colors.qualitative.Set2
):
    # Build plotting DataFrame and align clusters by position
    df = pd.DataFrame(
        umap_embedding[:, :dims],
        columns=[f"UMAP_{i+1}" for i in range(dims)]
    )
    df["Cluster"] = labels.values.astype(str)

    # 2D UMAP
    if dims == 2:
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color="Cluster",
            title=title,
            opacity=0.6,
            width=width,
            height=height,
            color_discrete_sequence=palette,
            category_orders={
                "Cluster": sorted(df["Cluster"].unique(), key=str)
            }
        )

        # Enable grid and enforce equal scaling (important for UMAP)
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            scaleanchor="x",
            scaleratio=1
        )

    # 3D UMAP
    elif dims == 3:
        fig = px.scatter_3d(
            df,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="Cluster",
            title=title,
            opacity=0.6,
            width=width,
            height=height,
            color_discrete_sequence=palette,
            category_orders={
                "Cluster": sorted(df["Cluster"].unique(), key=str)
            }
        )

    else:
        raise ValueError("dims must be 2 or 3")

    # Common layout styling
    fig.update_layout(
        legend_title_text="Segments",
        font=dict(size=14, family="Arial"),
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.show()



# --------------------------------------- METHOD COMPARISON GRAPHS --------------------------------------#

def plot_method_comparison(pca_coords, tsne_coords, umap_coords, labels, var_explained, figsize=(16, 5)):
    """ Plot side-by-side 2D projections from PCA, t-SNE, and UMAP for method comparison.
    Parameters:
        pca_coords : 2D PCA coordinates.
        tsne_coords : 2D t-SNE coordinates.
        umap_coords : 2D UMAP coordinates.
        labels : Cluster labels for coloring points.
        var_explained : Variance explained by PCA components.
        figsize : Figure size.
    Returns:
        Displays the comparison scatter plots.
    """

    # Create a figure with three panels to compare dimensionality reduction methods
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract unique cluster labels (robust to non-consecutive or unordered labels)
    clusters = sorted(np.unique(labels))

    # Use the Set2 colormap for soft, qualitative colouring of clusters
    # Accessed via the new Matplotlib API to avoid deprecation warnings
    cmap = mpl.colormaps["Set2"]

    # Assign one consistent color per cluster across all panels
    colors = {
        c: cmap(i / max(1, len(clusters) - 1))
        for i, c in enumerate(clusters)
    }

    # Helper function to draw a single scatter panel
    # This avoids repeating the same scatter logic for PCA, t-SNE and UMAP
    def scatter_panel(ax, X, title, xlabel, ylabel, equal_aspect=False):
        # Plot each cluster separately to ensure consistent coloring and legend entries
        for c in clusters:
            mask = labels == c
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=20,
                alpha=0.6,
                color=colors[c],
                edgecolors="none",
                label=f"Cluster {c}"
            )

        # Panel-specific titles and axis labels
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Light grid improves readability without cluttering the plot
        ax.grid(True, alpha=0.3)

        # Enforce equal scaling when geometric distances should be preserved
        if equal_aspect:
            ax.set_aspect("equal")

    # ----- PCA projection -----
    # Highlights global variance structure captured by the first two components
    scatter_panel(
        axes[0],
        pca_coords,
        "PCA: Global Structure",
        f"PC1 ({var_explained[0]:.1%})",
        f"PC2 ({var_explained[1]:.1%})"
    )

    # ----- t-SNE projection -----
    # Emphasises local neighbourhood relationships
    scatter_panel(
        axes[1],
        tsne_coords,
        "t-SNE: Local Structure",
        "t-SNE 1",
        "t-SNE 2"
    )

    # ----- UMAP projection -----
    # Provides a balance between local and global structure
    scatter_panel(
        axes[2],
        umap_coords,
        "UMAP: Balanced View",
        "UMAP 1",
        "UMAP 2",
        equal_aspect=True
    )

    # Create a single shared legend to avoid repetition across panels
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        title="Segments",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        framealpha=1,
        fontsize=12,
        markerscale=2
    )

    # Overall figure title summarising the comparison
    plt.suptitle(
        "Method Comparison: Same 7 Segments, Different Views",
        fontsize=14,
        fontweight="bold",
        y=1.05
    )

    # Adjust layout to make room for the shared legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.86, wspace=0.3)

    plt.show()

