import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from data_loader import load_data

# Load data once Streamlit starts running ✅
(
    flightsDB, customerDB, metaData,
    non_metric_features_FDB, metric_features_FDB, continuous_FDB,
    metric_features_CDB, continuous_CDB, non_metric_features_CDB
) = load_data()

# FUNCTIONS
def metric_histograms_CDB(customerDB, metric_features_CDB, continuous_CDB):
    sp_rows_CDB = 2
    sp_cols_CDB = ceil(len(metric_features_CDB) / sp_rows_CDB)

    sns.set_style("white")
    fig, axes = plt.subplots(sp_rows_CDB, 
                             sp_cols_CDB, 
                             figsize=(20, 10),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.2)

    for ax, feat in zip(axes.flatten(), metric_features_CDB):
        sns.histplot(customerDB[feat],
                     bins=10,
                     kde=(feat in continuous_CDB),
                     ax=ax,
                     edgecolor="black")
        ax.set_title(feat, fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("")

    fig.suptitle("Metric Variables' Histograms",
                 fontsize=25, fontweight="bold", y=1.05)

    return fig

# UI
st.title("EDA Dashboard")
st.subheader("Metric Variables' Histograms")

fig = metric_histograms_CDB(customerDB, metric_features_CDB, continuous_CDB)
st.pyplot(fig)
