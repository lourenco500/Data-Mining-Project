#--------------------------------------- LIBRARIES --------------------------------------#
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import clone



#--------------------------------------- BOXPLOTS --------------------------------------#
# Create boxplots function
def create_boxplots(df, n_cols=3, figsize=(20, 15)):
    """
    Create boxplots for all numeric columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        n_cols (int): Number of columns in the subplot grid.
        figsize (tuple): Size of the entire figure.

    Returns:
        None: Displays the boxplots.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        nrows = ceil(len(numeric_cols)/n_cols), ncols = n_cols,
        figsize=figsize,
        constrained_layout=True
    )

    for ax, feat in zip(axes.flatten(), numeric_cols):
        sns.boxplot(x=df[feat], ax=ax, color="#0088FF", medianprops={"color": "black", "linewidth": 2})
        ax.set_title(feat)
        ax.set_xlabel("")

    # esconder gráficos vazios
    for ax in axes.flatten()[len(numeric_cols):]:
        ax.set_visible(False)

    plt.suptitle("Metric Features — Boxplots", fontsize=22, fontweight="bold")
    plt.show()


# ----------------- HEATMAPS ----------------- #

# Function to create heatmap for correlation matrix of numeric columns
def create_heatmap(df, method, figsize=(10, 8)):
    """
    Creates a heatmap for the correlation matrix of numeric columns.
    
    Parameters:
        df (DataFrame): The dataset
        method (str): Correlation method to use (e.g., "pearson", "spearman", "kendall").
        numeric_cols (list): List of numeric columns to include in the correlation matrix.
    Returns:
        Displays a heatmap of the correlation matrix.
    """

    metric_cols = df.select_dtypes(include=['number']).columns # Select numeric columns
    # Calculate correlation matrix
    corr = df[metric_cols].corr(method=method).round(2)
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))  
    # Visualize correlation matrix
    fig = plt.figure(figsize=figsize)

    sns.heatmap(
    corr,
    mask=mask,                # hide upper triangle
    annot=True,               # show values
    cmap="coolwarm",          # divergent color map
    center=0,                 # center colormap in 0
    linewidths=0.5,           # lines between cells to help visualization
    vmin=-1, vmax=1,          # fix scale
    square=True               # make cells square-shaped
    )

    plt.title(f"Correlation Matrix ({method})", fontsize=14, pad=15)
    plt.tight_layout() # improve layout by reducing overlaps
    plt.show()


#--------------------------------------- SCALING --------------------------------------#

# Function to scale features using different scaling methods
def scaling_features(df, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Parameters:
        df (pd.DataFrame): The dataframe to fit and transform with the scaler.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        df (np.ndarray): The scaled dataframe to which the scaler is applied.
    """

    df = df.copy()

    metric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if method == 'minmax':
        # Scale and transform our data using MinMaxScaler[0,1]
        scaler = MinMaxScaler().fit(df[metric_cols])
        scaled_df = scaler.transform(df[metric_cols])
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to our data
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df[metric_cols])
        scaled_df = scaler.transform(df[metric_cols])
    elif method == 'standard':
        # Create a StandardScaler instance and fit to our data
        scaler = StandardScaler().fit(df[metric_cols])
        scaled_df = scaler.transform(df[metric_cols])
    else: 
        # Create a RobustScaler instance and fit to our data
        scaler = RobustScaler().fit(df[metric_cols])
        scaled_df = scaler.transform(df[metric_cols])

    # Replace the original metric columns with the scaled values
    df[metric_cols] = scaled_df

    return df, scaler



#--------------------------------------- CORRELATION PAIRS --------------------------------------#

def high_corr_pairs(df, corr_type, threshold=0.9):
    """ Function to find pairs of highly correlated features in a DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        corr_type (str): The type of correlation to compute ('pearson', 'spearman', 'kendall').
        threshold (float): The correlation threshold above which pairs are considered highly correlated.
    Returns:
        pd.Series: A Series containing pairs of features with correlation above the specified threshold.
    """

    df = df.copy()

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    corr = df[numeric_cols].corr(method = corr_type)
    # pegar só metade inferior para não repetir pares
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    corr_lower = corr.where(mask)

    # converter em lista de pares e filtrar só os acima do threshold
    high_corr = corr_lower.unstack().dropna()
    high_corr = high_corr[high_corr.abs() >= threshold]

    return high_corr.sort_values(ascending=False)



#--------------------------------------- SUM OF SQUARES --------------------------------------#

def get_ss(df_perspective):
    ss = np.sum(df_perspective.var() * (df_perspective.count() - 1))
    return ss 


#PODE NAO SER PRECISO. ACHO QUE PODE SER CALCULADO USANDO O GET_SS E O GET_SSW
# def get_ssb(df_perspective, label_col):
#     ssb_i = 0
#     for i in np.unique(df[label_col]):
#         df_ = df.loc[:, feats]
#         X_ = df_.values
#         X_k = df_.loc[df[label_col] == i].values
        
#         ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

#     ssb = np.sum(ssb_i)
    

#     return ssb


def get_ssw(df_perspective, label_col):

    feats = df_perspective.columns.tolist()
    feats_label = feats+[label_col]

    df_k = df_perspective[feats_label].groupby(by=label_col).apply(
        lambda col: get_ss(col, feats), 
        include_groups=False
        )

    return df_k.sum()

#--------------------------------------- R-SQUARED --------------------------------------#

def get_rsq(df_perspective, label_col):

    df_sst_ = get_ss(df_perspective)                 # get total sum of squares
    df_ssw_ = get_ssw(df_perspective, label_col)     # get ss within
    df_ssb_ = df_sst_ - df_ssw_                 # get ss between

    # r2 = ssb/sst 
    return (df_ssb_/df_sst_)




# Da aula
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt. 
    
    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        #create hierarchical clusterer
        cluster = AgglomerativeClustering(linkage=link_method, metric=dist, n_clusters=i)
        
        #get cluster labels
        hclabels = cluster.fit_predict(df[feats])
        
        # concat df with labels
        df_concat = pd.concat([df, pd.Series(hclabels, name='labels', index=df.index)], axis=1)  
        
        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, 'labels'))
        
    return np.array(r2)

# Da stora
def get_r2_scores(df, feats, clusterer, min_k=2, max_k=10):
    r2_clust = {}
    for n in range(min_k, max_k + 1):  # Ensure max_k is included
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df[feats])  # Use only the features
        df_concat = pd.concat([df, pd.Series(labels, name='labels', index=df.index)], axis=1)
        r2_clust[n] = get_rsq(df_concat, feats, 'labels')
    return r2_clust