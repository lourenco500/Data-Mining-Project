#--------------------------------------- LIBRARIES --------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder






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

    plt.suptitle(f"Metric Features — Boxplots", fontsize=15, fontweight="bold")
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
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'standard', and 'robust'.
    Returns:
        df (np.ndarray): The scaled dataframe to which the scaler is applied.
    """

    df = df.copy()

    metric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if method == 'minmax':
        # Scale and transform our data using MinMaxScaler[0,1]
        scaler = MinMaxScaler().fit(df[metric_cols])
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
def encoding_categorical_ohe(df, features_to_encode):
    """ Function to encode categorical features in a DataFrame using One-Hot Encoding.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
    Returns:
        pd.DataFrame: A DataFrame with categorical features encoded using One-Hot Encoding.
        scaler (OneHotEncoder): The fitted OneHotEncoder instance.
    """

    # Create a copy of the given DataFrame
    df_ohe = df.copy()

    # Initialize OneHotEncoder
    ohc = OneHotEncoder(sparse_output=False)

    # Apply OneHotEncoding to the categorical columns
    ohc_feat = ohc.fit_transform(df_ohe[features_to_encode])

    # Get the feature names after encoding
    ohc_feat_names = ohc.get_feature_names_out(features_to_encode)

    # Convert the encoded features to a DataFrame
    ohc_features = pd.DataFrame(ohc_feat, index=df_ohe.index, columns=ohc_feat_names)

    # Concatenate the new DataFrame with the original one
    df_ohe = pd.concat([df_ohe, ohc_features], axis=1)

    # Update the final DataFrame
    df = df_ohe.copy()

    # Dropping categorical Variables
    df = df.drop(columns = features_to_encode)

    return df, ohc_feat_names




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







