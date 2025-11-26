
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_boxplots(df, numeric_cols, n_cols=2, figsize=(12, 8)):
    n_rows = ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    axes = np.array(axes).flatten()
    
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    
    # Esconde eixos extras
    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()
