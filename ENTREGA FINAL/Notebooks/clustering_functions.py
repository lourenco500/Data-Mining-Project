import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import clone
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import silhouette_score


#--------------------------------------- SUM OF SQUARES --------------------------------------#

def get_ss(df, feats):

    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    
    return ss 


def get_ssb(df, feats, label_col):
    
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb


def get_ssw(df, feats, label_col):

    feats_label = feats+[label_col]

    df_k = df[feats_label].groupby(by=label_col).apply(
        lambda col: get_ss(col, feats), 
        include_groups=False
        )

    return df_k.sum()



#--------------------------------------- R-SQUARED --------------------------------------#

def get_rsq(df, feats, label_col):
    df_sst_ = get_ss(df, feats)                 # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)     # get ss within
    df_ssb_ = df_sst_ - df_ssw_                 # get ss between

    # r2 = ssb/sst 
    return (df_ssb_/df_sst_)

def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """Function to compute R² values for hierarchical clustering over a range of cluster numbers.
    Parameters:
        df : DataFrame containing the data to be clustered.
        link_method : Linkage method to be used in hierarchical clustering (e.g., 'ward', 'complete', 'average').
        max_nclus : Maximum number of clusters to evaluate.
        min_nclus : Minimum number of clusters to evaluate (default is 1).
        dist : Distance metric to be used (default is 'euclidean').
    Returns: 
        Array of R² values corresponding to each number of clusters evaluated.
    """
    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        
        cluster = AgglomerativeClustering(linkage=link_method, metric=dist, n_clusters=i)
        
        #get cluster labels
        hclabels = cluster.fit_predict(df[feats])
        
        # concat df with labels
        df_concat = pd.concat([df, pd.Series(hclabels, name='labels', index=df.index)], axis=1)  
        
        
        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, 'labels'))
        
    return np.array(r2)


# --------------------------------------- R-SQUARED PER VARIABLE --------------------------------------#

def get_ss_variables(df):
    """
    Get the Sum of Squares (SS) for each variable
    """
    return df.var() * (df.count() - 1)


def r2_variables(df, labels):
    """
    Get the R² for each variable
    """
    # Total Sum of Squares
    sst_vars = get_ss_variables(df)

    # Within-cluster Sum of Squares
    ssw_vars = (
        df
        .groupby(labels)
        .apply(get_ss_variables, include_groups=False)
        .sum(axis=0)
    )

    return 1 - ssw_vars / sst_vars


#--------------------------------------- SOM HEXAGON PLOTTING --------------------------------------#

# Function to plot hexagons 
def plot_hexagons_ax(som, ax, matrix_vals, colornorm, label="", cmap=plt.cm.Grays, annot=False):
    """ Function to plot hexagons on a given axis for a Self-Organizing Map (SOM).
    Parameters:
        som : Trained SOM object.
        ax : Matplotlib axis to plot on.
        matrix_vals : 2D array of values to be represented in the hexagons.
        colornorm : Normalization object for color mapping.
        label : Title label for the plot.
        cmap : Colormap to use for the hexagons.
        annot : Boolean indicating whether to annotate hexagons with their values.
    """

    # draw hexagons
    for i in range(matrix_vals.shape[0]):
        for j in range(matrix_vals.shape[1]):
            x, y = som.convert_map_to_euclidean((i, j))
            val = matrix_vals[i, j]

            ax.add_patch(
                RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=np.sqrt(1 / 3),
                    facecolor=cmap(colornorm(val)),
                    edgecolor="white",
                    linewidth=0.5
                )
            )
            # add annotation
            if annot:
                txt = int(val) if val == int(val) else np.round(val, 2)
                ax.text(x, y, txt, ha="center", va="center", fontsize="x-small")

    # styling
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize = 10, fontweight="bold")
    ax.margins(0.05)

    # create colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)

    # create colorbar
    sm = plt.cm.ScalarMappable(norm=colornorm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=7)



#--------------------------------------- SOM VISUALIZATION HELPERS --------------------------------------#

def get_index_matrix(nx, ny):
    """ Function to create a matrix with the indices of each unit in the SOM grid.
    Parameters:
        nx : Number of units in the x-direction.
        ny : Number of units in the y-direction.
    Returns:
        2D array with the indices of each unit in the SOM grid.
    """

    # create meshgrid
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    # Initialize matrix to hold index strings
    mx_index = np.full((nx,ny),str)

    # Fill matrix with index strings
    for i, j in zip(xv,yv):    
        for ii, jj  in zip(i,j):
            ii = int(ii)
            jj = int(jj)
            mx_index[ii,jj] = f"({ii},{jj})"

    return mx_index


# --------------------------------------- CLUSTERING MODELS COMPARISON --------------------------------------#

def compare_clustering_models(datasets_dict, feature_cols, label_col='cluster_labels_VE', title='Clustering Models Comparison'):
    """ Function to compare clustering models using R² and Silhouette scores.
    Parameters:
        datasets_dict : Dictionary where keys are model names and values are DataFrames with clustering results.
        feature_cols : List of feature column names used for clustering.
        label_col : Name of the column containing cluster labels (default is 'cluster_labels_VE').
        title : Title for the comparison plot (default is 'Clustering Models Comparison').
    Returns:
        Displays a bar plot comparing R² and Silhouette scores for each clustering model.
    """
    # create lists to store results
    models, r2_scores, silhouette_scores, n_clusters_list = [], [], [], []

    # iterate over datasets
    for model_name, df in datasets_dict.items():

        # remove noise points if present
        if (df[label_col] == -1).any():
            df_eval = df[df[label_col] != -1]
        else:
            df_eval = df

        # nº of clusters (without noise)
        n_clusters = df_eval[label_col].nunique()
        n_clusters_list.append(n_clusters)

        # Silhouette
        X = df_eval[feature_cols]
        labels = df_eval[label_col]

        if labels.nunique() > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = np.nan

        # R²
        r2 = get_rsq(df_eval, feature_cols, label_col)

        # append results
        models.append(model_name)
        r2_scores.append(r2)
        silhouette_scores.append(sil)

    # plot
    x = np.arange(len(models))
    width = 0.20

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, r2_scores, width, label='R²')
    plt.bar(x + width/2, silhouette_scores, width, label='Silhouette')

    # Custom x-tick labels with number of clusters
    x_labels = [f"{m}\n(k={k})" for m, k in zip(models, n_clusters_list)]
    plt.xticks(x, x_labels, rotation=0)

    plt.ylabel('Metric Value')
    plt.title(title, fontsize=14, fontweight="bold", pad=15)

    # Add legend below the plot
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False
    )

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()