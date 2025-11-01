import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from data_loader import load_data
from graph_functions import boxplots_CDB, location_scatter_CDB
import numpy as np

# -------------------------
# Utility helpers
# -------------------------
def clear_all_filters():
    """Clear session state and rerun to reset widgets."""
    st.session_state.clear()
    st.experimental_rerun()

def safe_slider(label, min_val, max_val, key=None):
    """Create a slider only when there's a valid range; otherwise show a message and return None."""
    if pd.isna(min_val) or pd.isna(max_val):
        st.sidebar.write(f"⚠️ Not enough data to filter `{label}`")
        return None
    try:
        imin, imax = int(min_val), int(max_val)
    except Exception:
        try:
            imin, imax = float(min_val), float(max_val)
        except Exception:
            st.sidebar.write(f"⚠️ Cannot create slider for `{label}`")
            return None
    if imin < imax:
        return st.sidebar.slider(label, imin, imax, (imin, imax), key=key)
    else:
        st.sidebar.write(f"⚠️ Not enough variation to filter `{label}`")
        return None

# -------------------------
# Load data
# -------------------------
(
    flightsDB, customerDB, metaData,
    non_metric_features_FDB, metric_features_FDB, continuous_FDB,
    metric_features_CDB, continuous_CDB, non_metric_features_CDB
) = load_data()

# Remove Unnamed: 0 globally if present and from feature lists
for unwanted in ["Unnamed: 0"]:
    if unwanted in customerDB.columns:
        customerDB.drop(columns=[unwanted], inplace=True)
    if unwanted in metric_features_CDB:
        try:
            metric_features_CDB.remove(unwanted)
        except ValueError:
            pass
    if unwanted in continuous_CDB:
        try:
            continuous_CDB.remove(unwanted)
        except ValueError:
            pass

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Customer Explorer", "Flights Explorer", "Insights & Graphs"])

# -------------------------
# PAGE: Customer Explorer
# -------------------------
if page == "Customer Explorer":
    st.title("👤 Customer Database Explorer")
    df = customerDB.copy()

    st.sidebar.subheader("Filters")
    if st.sidebar.button("Clear All Filters"):
        clear_all_filters()

    # Convert date columns
    for date_col in ["EnrollmentDateOpening", "CancellationDate"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Date filters
    for col, label in [("EnrollmentDateOpening", "Enrollment Date"), ("CancellationDate", "Cancellation Date")]:
        if col in df.columns:
            min_d, max_d = df[col].min(), df[col].max()
            if not pd.isna(min_d) and not pd.isna(max_d):
                date_range = st.sidebar.date_input(label, (min_d.date(), max_d.date()), key=f"C_{col}")
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    start = pd.to_datetime(date_range[0])
                    end = pd.to_datetime(date_range[1])
                    df = df[df[col].between(start, end)]

    # Dropdown filters (safe checks for existence)
    filter_cols = {
        "City": "City",
        "Province or State": "Province or State",
        "Gender": "Gender",
        "Education": "Education",
        "Location Code": "Location Code",
        "Marital Status": "Marital Status",
        "Loyalty Status": "LoyaltyStatus",
        "Enrollment Type": "EnrollmentType"
    }
    for label, col in filter_cols.items():
        if col in df.columns:
            opts = sorted(df[col].dropna().unique().tolist())
            sel = st.sidebar.multiselect(label, opts, key=f"C_{col}")
            if sel:
                df = df[df[col].isin(sel)]

    # Income slider
    if "Income" in df.columns:
        rng = safe_slider("Income", df["Income"].min(), df["Income"].max(), key="C_Income")
        if rng is not None:
            df = df[df["Income"].between(rng[0], rng[1])]

    # CLV slider
    if "Customer Lifetime Value" in df.columns:
        rng = safe_slider("Customer Lifetime Value", df["Customer Lifetime Value"].min(), df["Customer Lifetime Value"].max(), key="C_CLV")
        if rng is not None:
            df = df[df["Customer Lifetime Value"].between(rng[0], rng[1])]

    # If empty after filters
    if df.empty:
        st.warning("⚠️ No records match your filters. Clear filters or broaden your selection.")
        st.stop()

    # ---------- KPI cards (Customer)
    kpi1, kpi2 = st.columns(2)
    # Total Customer Lifetime Value
    total_clv = df["Customer Lifetime Value"].sum() if "Customer Lifetime Value" in df.columns else np.nan
    # City mode
    city_mode = df["City"].mode().iloc[0] if ("City" in df.columns and not df["City"].dropna().empty) else "—"

    kpi1.metric("Total Customer Lifetime Value", f"{total_clv:,.0f}")
    kpi2.metric("Most common City", f"{city_mode}")

    st.markdown("### Filtered Customer Data")
    st.dataframe(df, use_container_width=True, height=700)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name="filtered_customers.csv", mime="text/csv")

# -------------------------
# PAGE: Flights Explorer
# -------------------------
elif page == "Flights Explorer":
    st.title("✈️ Flight Records Explorer")
    df = flightsDB.copy()

    st.sidebar.subheader("Filters")
    if st.sidebar.button("Clear All Filters"):
        clear_all_filters()

    # Date filter
    if "YearMonthDate" in df.columns:
        df["YearMonthDate"] = pd.to_datetime(df["YearMonthDate"], errors="coerce")
        min_d, max_d = df["YearMonthDate"].min(), df["YearMonthDate"].max()
        if not pd.isna(min_d) and not pd.isna(max_d):
            date_range = st.sidebar.date_input("Year / Month", (min_d.date(), max_d.date()), key="F_Date")
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                df = df[df["YearMonthDate"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]

    # Numeric sliders (only)
    slider_cols = [
        "NumFlights",
        "NumFlightsWithCompanions",
        "DistanceKM",
        "PointsAccumulated",
        "PointsRedeemed",
        "DollarCostPointsRedeemed"
    ]
    for col in slider_cols:
        if col in df.columns:
            rng = safe_slider(col, df[col].min(), df[col].max(), key=f"F_{col}")
            if rng is not None:
                df = df[df[col].between(rng[0], rng[1])]

    if df.empty:
        st.warning("⚠️ No flight records match your filters. Clear filters or broaden your selection.")
        st.stop()

    # --------- KPI cards (Flights)
    f1, f2, f3 = st.columns(3)
    total_flights = int(df["NumFlights"].sum()) if "NumFlights" in df.columns else 0
    total_accum = int(df["PointsAccumulated"].sum()) if "PointsAccumulated" in df.columns else 0
    total_redeemed = int(df["PointsRedeemed"].sum()) if "PointsRedeemed" in df.columns else 0


    f1.metric("Total Number of Flights", f"{total_flights:,}")
    f2.metric("Total Points Accumulated", f"{total_accum:,}")
    f3.metric("Total Points Redeemed", f"{total_redeemed:,}")


    st.markdown("### Filtered Flight Data")
    st.dataframe(df, use_container_width=True, height=700)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name="filtered_flights.csv", mime="text/csv")

# -------------------------
# PAGE: Insights & Graphs
# -------------------------
elif page == "Insights & Graphs":
    st.title("📊 Customer Insights Dashboard")

    st.sidebar.subheader("Insights Controls (CustomerDB)")

    # Available numeric metrics (safe)
    available_metrics = [c for c in metric_features_CDB if c in customerDB.columns]
    if not available_metrics:
        st.warning("No numeric features found in CustomerDB for plotting.")
    else:
        selected = st.sidebar.multiselect(
            "Select metric features to plot (boxplots)",
            options=available_metrics,
            default=available_metrics[:8]
        )

        if selected:
            # Auto layout rows: choose 2 rows if <=8 else more rows to keep readable
            n = len(selected)
            rows = 2 if n <= 8 else int(np.ceil(n / 4))
            try:
                fig_box = boxplots_CDB(customerDB, selected, rows=rows)
                st.pyplot(fig_box)
            except Exception as e:
                st.error(f"Failed to build boxplots: {e}")

    # Location scatter (plot ALL points)
    st.sidebar.markdown("---")
    show_location = st.sidebar.checkbox("Show customer location scatter", value=True)
    show_trend = st.sidebar.checkbox("Show trend line on location scatter", value=False)

    if show_location:
        if ("Longitude" in customerDB.columns) and ("Latitude" in customerDB.columns):
            try:
                fig_loc = location_scatter_CDB(customerDB, lon_col="Longitude", lat_col="Latitude", show_trend=show_trend)
                st.pyplot(fig_loc)
            except Exception as e:
                st.error(f"Failed to draw location scatter: {e}")
        else:
            st.info("Longitude/Latitude columns not found in CustomerDB; cannot display location scatter.")
