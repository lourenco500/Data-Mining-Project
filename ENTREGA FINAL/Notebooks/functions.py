#--------------------------------------- LIBRARIES --------------------------------------#
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



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
    Args:
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
        scaled_df = MinMaxScaler().fit_transform(df[metric_cols])
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to our data
        scaled_df = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[metric_cols])
    elif method == 'standard':
        # Create a StandardScaler instance and fit to our data
        scaled_df = StandardScaler().fit_transform(df[metric_cols])
    else: 
        # Create a RobustScaler instance and fit to our data
        scaled_df = RobustScaler().fit_transform(df[metric_cols])

    # Replace the original metric columns with the scaled values
    df[metric_cols] = scaled_df

    return df



# ------------------ HIGHLY CORRELATED FEATURES ------------------ #

# Remove one feature from each pair of highly correlated numerical features
def remove_correlated_features(df, method='spearman', high_threshold = 0.9, low_threshold=0.01, return_summary=False):
    """ Removes one feature from each pair of highly correlated numerical features based on training data only.
        Parameters:
            df: DataFrame containing the features.
            method: Correlation method to use (e.g., "pearson", "spearman").
            threshold: Correlation threshold above which one feature from the pair will be removed.
            return_summary: If True, prints a summary of the features removed.
        Returns:
            df: DataFrame with highly correlated features removed.
    """
    

    df = df.copy()
    # Select only numerical columns
    num_cols = df.select_dtypes(include='number').columns
    
    # Compute absolute correlation matrix
    corr_matrix = df[num_cols].corr(method='spearman').abs()
    
    # Create mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > high_threshold) and any(upper[column] < low_threshold)]

    # Identify features to keep
    cols_to_keep = [col for col in df.columns if col not in to_drop]

    # Apply to df_to_apply
    df = df[cols_to_keep]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(num_cols) - len(to_drop)}")
        print(f"Features selected: {cols_to_keep}")
        print(f"Nº Features eliminated: {len(to_drop)}")
    
    return df



# ------------------ LOW CORRELATED FEATURES ------------------ #

# Remove one feature from each pair of low correlated numerical features
def remove_low_correlated_features(df, method='spearman', threshold = 0.9, return_summary=False):
    """ Removes one feature from each pair of highly correlated numerical features based on training data only.
        Parameters:
            df: DataFrame containing the features.
            method: Correlation method to use (e.g., "pearson", "spearman").
            threshold: Correlation threshold under which one feature from the pair will be removed.
            return_summary: If True, prints a summary of the features removed.
        Returns:
            df: DataFrame with low correlated features removed.
    """
    

    df = df.copy()
    # Select only numerical columns
    num_cols = df.select_dtypes(include='number').columns
    
    # Compute absolute correlation matrix
    corr_matrix = df[num_cols].corr(method='spearman').abs()
    
    # Create mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper.columns if any(upper[column] < threshold)]

    # Identify features to keep
    cols_to_keep = [col for col in df.columns if col not in to_drop]

    # Apply to df_to_apply
    df = df[cols_to_keep]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(num_cols) - len(to_drop)}")
        print(f"Features selected: {cols_to_keep}")
        print(f"Nº Features eliminated: {len(to_drop)}")
    
    return df
