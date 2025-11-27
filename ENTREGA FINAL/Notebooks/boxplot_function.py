
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_boxplots(df, numeric_cols, n_cols=3, figsize=(20, 10)):
    # Prepare figure. Create individual axes where each histogram will be placed
    fig, axes = plt.subplots(ceil(len(numeric_cols) / n_cols), 
                            n_cols, 
                            figsize,
                            constrained_layout=True) # Adjust automatically spacing between subplot and labels 

    # Increase vertical space between rows
    fig.set_constrained_layout_pads(h_pad=.2)

    # Plot data
    # Iterate across axes objects and associate each box plot:
    for ax, feat in zip(axes.flatten(), numeric_cols): # "[2:]" to skip Year and Month features
        sns.boxplot(x=df[feat], ax=ax,
                    # Set inside color to blue
                    color="#0062FF",
                    # Set line colors to black
                    boxprops=dict(edgecolor="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    medianprops=dict(color="black", linewidth=3),
                    flierprops=dict(markeredgecolor="black"))
        
        # Put grid with low opacity and dashed line so it's visible, but not distracting
        ax.grid(True, linestyle="--", alpha=0.4)

        # Put title of each graph with bigger font
        ax.set_title(feat, fontsize=18)

        # Hide x label since we already have the boxlplot title at the top
        ax.set_xlabel("")
        
    # Layout
    # Add title and make it bigger
    plt.suptitle("Metric features' Box Plots", 
                fontsize = 25, fontweight="bold")
