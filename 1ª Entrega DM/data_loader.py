import pandas as pd
import os

def load_data():
    """Load FlightsDB, CustomerDB and Metadata CSVs with correct paths."""

    FILE_DIR = os.path.dirname(__file__)  # Folder where this .py file exists
    DATA_DIR = os.path.join(FILE_DIR, "data")  # Data folder next to app.py

    # Build reliable absolute paths
    flights_path = os.path.join(DATA_DIR, "DM_AIAI_FlightsDB.csv")
    customer_path = os.path.join(DATA_DIR, "DM_AIAI_CustomerDB.csv")
    metadata_path = os.path.join(DATA_DIR, "DM_AIAI_Metadata.csv")

    # Load CSVs
    flightsDB = pd.read_csv(flights_path, sep=",", index_col="Loyalty#")
    customerDB = pd.read_csv(customer_path, sep=",", index_col="Loyalty#")
    metaData = pd.read_csv(metadata_path, sep=";", header=None)

    # Feature lists
    non_metric_features_FDB = ["YearMonthDate"]
    metric_features_FDB = flightsDB.columns.drop(non_metric_features_FDB).to_list()
    continuous_FDB = ["DistanceKM", "PointsAccumulated", "PointsRedeemed", "DollarCostPointsRedeemed"]

    metric_features_CDB = customerDB.select_dtypes(include=['number']).columns.tolist()
    continuous_CDB = ["Latitude", "Longitude", "Income", "Customer Lifetime Value"]
    non_metric_features_CDB = customerDB.columns.drop(metric_features_CDB).to_list()

    # categorical_features_CDB = customerDB.select_dtypes(include= 'object', exclude='datetime').columns.tolist()
    # discrete_features_CDB = customerDB[["EnrollmentYear", "CancellationYear"]].columns.tolist()

    return (
        flightsDB, customerDB, metaData,
        non_metric_features_FDB, metric_features_FDB, continuous_FDB,
        metric_features_CDB, continuous_CDB, non_metric_features_CDB
    )
