#--------------------------------------- LIBRARIES --------------------------------------#
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import clone
from sklearn.metrics import silhouette_score


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

    axes = np.atleast_1d(axes)

    for ax, feat in zip(axes.flatten(), numeric_cols):
        sns.boxplot(x=df[feat], ax=ax, color="#0088FF", medianprops={"color": "black", "linewidth": 2})
        ax.set_title(feat)
        ax.set_xlabel("")

    # hide unused axes
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
    Returns:
        Displays a heatmap of the correlation matrix.
    """

    # Select numeric columns
    cols = df.select_dtypes(include=['number']).columns.tolist()

    # Calculate correlation matrix
    corr = df[cols].corr(method=method).round(2)
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


# --------------------------------------- OUTLIER TREATMENT --------------------------------------#

def treat_outliers_cap(df, cols, threshold = 0.999):
    """ Function to treat outliers in specified columns of a DataFrame by capping them at a given quantile threshold.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        cols (list): List of column names to treat for outliers.
        threshold (float): The quantile threshold to use for capping outliers (default is 0.999).
    Returns:
        pd.DataFrame: A DataFrame with outliers treated in the specified columns.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    for col in cols:
        # Calculate the quantile limit for the specified threshold
        limit = df[col].quantile(threshold)

        # Cap the outliers at the calculated limit
        df = df[df[col] <= limit]

    return df



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


#--------------------------------------- ONE-HOT ENCODING --------------------------------------#

# Function to encode categorical features using One-Hot Encoding
def encoding_categorical_ohe(df):
    """ Function to encode categorical features in a DataFrame using One-Hot Encoding.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
    Returns:
        pd.DataFrame: A DataFrame with categorical features encoded using One-Hot Encoding.
        scaler (OneHotEncoder): The fitted OneHotEncoder instance.
    """

    # Create a copy of the given DataFrame
    df_ohe = df.copy()

    # Select categorical features (object and category columns)
    categorical_features = df_ohe.select_dtypes(include=["object", "category"]).columns

    # Initialize OneHotEncoder
    ohc = OneHotEncoder(sparse_output=False)

    # Apply OneHotEncoding to the categorical columns
    ohc_feat = ohc.fit_transform(df_ohe[categorical_features])

    # Get the feature names after encoding
    ohc_feat_names = ohc.get_feature_names_out(categorical_features)

    # Convert the encoded features to a DataFrame
    ohc_features = pd.DataFrame(ohc_feat, index=df_ohe.index, columns=ohc_feat_names)

    # Concatenate the new DataFrame with the original one
    df_ohe = pd.concat([df_ohe, ohc_features], axis=1)

    # Update the final DataFrame
    df = df_ohe.copy()

    # Dropping categorical Variables
    df = df.drop(columns = categorical_features)

    return df, ohc




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

    # Select numeric columns
    cols = df.select_dtypes(include=['number']).columns.tolist()

    corr = df[cols].corr(method = corr_type)
    # Take only the lower triangle of the correlation matrix
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    corr_lower = corr.where(mask)

    # Convert the correlation matrix to a Series of pairs and filter by threshold
    high_corr = corr_lower.unstack().dropna() # Drop correlations with NaN values
    high_corr = high_corr[high_corr.abs() >= threshold] # Keep only pairs with correlation above the threshold

    return high_corr.sort_values(ascending=False)



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

def get_r2_scores(df, feats, method, min_nclus=1, max_nclus=10):
    r2_clust = []
    for i in range(min_nclus, max_nclus + 1):  # Ensure max_k is included
        clust = clone(method).set_params(n_clusters=i)
        labels = clust.fit_predict(df[feats])  # Use only the features
        df_concat = pd.concat([df, pd.Series(labels, name='labels', index=df.index)], axis=1)
        r2_clust.append(get_rsq(df_concat, feats, 'labels'))
    return r2_clust

def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
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



## SOM Visualization Helpers

def get_index_matrix(nx, ny):
    # Given nx=2, ny=3
    # This will return a matrix whose contents are the index values:
    # 
    # (0,0) (0,1) (0,2)
    # (1,0) (1,1) (1,2)

    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    xv, yv = np.meshgrid(x, y, indexing='xy')


    mx_index = np.full((nx,ny),str)

    for i, j in zip(xv,yv):    
        for ii, jj  in zip(i,j):
            ii = int(ii)
            jj = int(jj)
            mx_index[ii,jj] = f"({ii},{jj})"

    return mx_index



# Visualize RGB SOM as colored grid

def plot_rgb_matrix(som_matrix, som, ax, annot_mx=None):

    som_x, som_y = som.get_weights().shape[:2]
    
    for i in range(som_x):
        for j in range(som_y):
            # Get RGB color for this unit (ensure values in 0-1 range)
            color = np.clip(som_matrix[i, j], 0, 1)
            
            # Convert grid to hexagonal coordinates
            hex_coord = som.convert_map_to_euclidean((i, j))
            
            center = [hex_coord[0], hex_coord[1]]
            
            # Draw hexagon with learned color
            hexagon = RegularPolygon(
                center, 
                numVertices=6, 
                radius=np.sqrt(1/3),
                facecolor=color, 
                edgecolor='white', 
                linewidth=2
            )
            ax.add_patch(hexagon)
            
            if "None" in str((type(annot_mx))):
                pass 
            else:
                
                annot_val = annot_mx[i,j]
                if "str" in str(type(annot_val)):
                    pass 
                else:
                    if int(annot_val) == annot_val:
                        annot_val = int(annot_val)
                
                ax.text(center[0], center[1], annot_val, 
                        ha='center', va='center',
                        fontsize='x-small')
    

    
    ax.set_xlim(-1, som_x-.5)
    ax.set_ylim(-1, som_y)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax




# --------------------------------------- CLUSTERING MODELS COMPARISON --------------------------------------#

def compare_clustering_models(datasets_dict, feature_cols, label_col='cluster_labels_VE'):

    models, r2_scores, silhouette_scores = [], [], []

    for model_name, df in datasets_dict.items():

        # remove noise points if present
        if (df[label_col] == -1).any():
            df_eval = df[df[label_col] != -1]
        else:
            df_eval = df

        # Silhouette
        X = df_eval[feature_cols]
        labels = df_eval[label_col]

        if labels.nunique() > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = np.nan

        # R²
        r2 = get_rsq(df_eval, feature_cols, label_col)

        models.append(model_name)
        r2_scores.append(r2)
        silhouette_scores.append(sil)

    # plot
    x = np.arange(len(models))
    width = 0.20

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, r2_scores, width, label='R²')
    plt.bar(x + width/2, silhouette_scores, width, label='Silhouette')

    plt.xticks(x, models, rotation=45)
    plt.ylabel('Metric Value')
    plt.title('Clustering Models Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
