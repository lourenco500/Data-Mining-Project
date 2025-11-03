import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
from data_loader import load_data
from graph_functions import boxplots_CDB, location_scatter_CDB


if st.session_state.get("__filters_cleared__", False):
    # Reset the flag
    st.session_state["__filters_cleared__"] = False
    # Streamlit will rerun and recreate widgets properly
    st.experimental_rerun()

# -------------------------
# Utility helpers
# -------------------------
def clear_all_filters():
    """Reset all filters safely."""
    for key in list(st.session_state.keys()):
        if key.startswith("C_") or key.startswith("F_"):
            # Multiselects: empty list
            if "Date" not in key:
                st.session_state[key] = []
            # Sliders: None
            else:
                st.session_state[key] = None

    # Additionally, for numeric sliders, reset to None so safe_slider uses full range
    for key in st.session_state.keys():
        if key.startswith("F_") and key not in ["F_Date"]:
            st.session_state[key] = None
        if key.startswith("C_") and "Date" not in key:
            st.session_state[key] = []



def safe_slider(label, min_val, max_val, key=None):
    """Create a slider safely. Handles session_state None values."""
    
    # Validate min/max
    if min_val is None or max_val is None or pd.isna(min_val) or pd.isna(max_val):
        st.sidebar.write(f"⚠️ Not enough data to filter `{label}`")
        return None

    try:
        imin, imax = float(min_val), float(max_val)
    except Exception:
        st.sidebar.write(f"⚠️ Cannot create slider for `{label}`")
        return None

    if imin >= imax:
        st.sidebar.write(f"⚠️ Not enough variation to filter `{label}`")
        return None

    # Retrieve current value from session_state
    val = st.session_state.get(key)

    # If val is invalid or contains None, reset to full range
    if (
        val is None
        or not isinstance(val, (tuple, list))
        or len(val) != 2
        or val[0] is None
        or val[1] is None
    ):
        val = (imin, imax)

    # Ensure both entries are floats
    try:
        val = (float(val[0]), float(val[1]))
    except Exception:
        val = (imin, imax)

    # Update session_state so next render uses this valid value
    st.session_state[key] = val

    return st.sidebar.slider(label, imin, imax, val, key=key)





# -------------------------
# Load data
# -------------------------
(
    flightsDB, customerDB, metaData,
    non_metric_features_FDB, metric_features_FDB, continuous_FDB,
    metric_features_CDB, continuous_CDB, non_metric_features_CDB
) = load_data()

# Remove Unnamed: 0 globally if present
if "Unnamed: 0" in customerDB.columns:
    customerDB.drop(columns=["Unnamed: 0"], inplace=True)

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Customer Explorer", "Flights Explorer", "Insights & Graphs"])

# -------------------------
# PAGE: Customer Explorer
# -------------------------
# At the very top of Customer Explorer page
st.session_state["__customer_df__"] = customerDB.copy()
df = st.session_state["__customer_df__"]

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
                # Use session_state to detect user selection
                ss_key = f"C_{col}"

                # Initialize session_state on first run
                if ss_key not in st.session_state:
                    st.session_state[ss_key] = None  # no filter applied yet

                # Show date input with current session_state or empty
                date_range = st.sidebar.date_input(
                    label,
                    value=st.session_state[ss_key] or (min_d.date(), max_d.date()),
                    key=ss_key
                )

                # Only filter df if user has changed the dates
                if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    # Only filter if session_state has a value different from None
                    if st.session_state[ss_key] is not None:
                        start = pd.to_datetime(date_range[0])
                        end = pd.to_datetime(date_range[1])
                        df = df[df[col].between(start, end)]


    # Dropdown filters
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

    # Dropdown filters
    for label, col in filter_cols.items():
        if col in df.columns:
            opts = df[col].dropna().unique()
            opts = sorted(opts.tolist()) if opts is not None else []

            # Get stored selection, default to empty list
            ss_key = f"C_{col}"
            sel_default = st.session_state.get(ss_key, [])

            # Only keep defaults that exist in options
            sel_default = [v for v in sel_default if v in opts]

            # Save back to session_state
            st.session_state[ss_key] = sel_default

            sel = st.sidebar.multiselect(
                label,
                options=opts,
                default=sel_default,
                key=ss_key
            )

            if sel:
                df = df[df[col].isin(sel)]




    # Numeric sliders
    # Income slider
    if "Income" in df.columns:
        rng = safe_slider("Income", df["Income"].min(), df["Income"].max(), key="C_Income")
        if rng is None or len(rng) != 2:
            rng = (df["Income"].min(), df["Income"].max())
        df = df[df["Income"].between(rng[0], rng[1])]

    # Customer Lifetime Value slider
    if "Customer Lifetime Value" in df.columns:
        rng = safe_slider("Customer Lifetime Value", df["Customer Lifetime Value"].min(), df["Customer Lifetime Value"].max(), key="C_CLV")
        if rng is None or len(rng) != 2:
            rng = (df["Customer Lifetime Value"].min(), df["Customer Lifetime Value"].max())
        df = df[df["Customer Lifetime Value"].between(rng[0], rng[1])]




    # Empty check
    if df.empty:
        st.warning("⚠️ No records match your filters. Clear filters or broaden your selection.")
        st.stop()

    # ---------- KPI cards (Customer)
    kpi1, kpi2 = st.columns(2)
    total_clv = df["Customer Lifetime Value"].sum() if "Customer Lifetime Value" in df.columns else np.nan
    city_mode = df["City"].mode().iloc[0] if ("City" in df.columns and not df["City"].dropna().empty) else "—"
    kpi1.metric("Total Customer Lifetime Value", f"{total_clv:,.0f}")
    kpi2.metric("Most common City", f"{city_mode}")

    st.markdown("### Filtered Customer Data")
    st.dataframe(df, use_container_width=True, height=700)

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

    # Numeric sliders
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

    # KPI cards (Flights)
    f1, f2 = st.columns(2)
    total_accum = int(df["PointsAccumulated"].sum()) if "PointsAccumulated" in df.columns else 0
    total_redeemed = int(df["PointsRedeemed"].sum()) if "PointsRedeemed" in df.columns else 0

    f1.metric("Total Points Accumulated", f"{total_accum:,}")
    f2.metric("Total Points Redeemed", f"{total_redeemed:,}")

    st.markdown("### Filtered Flight Data")
    st.dataframe(df, use_container_width=True, height=700)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name="filtered_flights.csv", mime="text/csv")

# -------------------------
# PAGE: Insights & Graphs
# -------------------------
elif page == "Insights & Graphs":
    st.title("📊 Customer Insights Dashboard")

    st.sidebar.subheader("Insights Controls (CustomerDB)")

    # Numeric metric selection
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
            try:
                fig_box = boxplots_CDB(customerDB, selected)
                st.pyplot(fig_box)
            except Exception as e:
                st.error(f"Failed to build boxplots: {e}")

    # Location scatter
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
