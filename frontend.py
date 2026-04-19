# Complete Streamlit Dashboard for Electricity Theft Detection
# -----------------------------------------------------------
# Stable production-ready version
# Includes:
# - Smooth progress bar
# - Processing time display
# - Dataset summary metrics
# - Risk filter
# - Top 10 consumers
# - Risk & Reason charts
# - Confidence histogram
# - Trend visualization (fixed bug)
# - Download report

import streamlit as st
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import time

# Import backend function
from prediction import predict_from_csv

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Theft Detection Dashboard",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("⚡ Electricity Theft Detection Dashboard")

st.markdown(
    """
    Upload electricity consumption data to identify potential theft locations.
    This dashboard provides risk analysis, financial impact, and operational insights.
    """
)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    st.success("File uploaded successfully")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    try:

        if st.button("Run Theft Detection"):

            # ------------------------------
            # Start timer
            # ------------------------------

            start_time = time.time()

            # ------------------------------
            # Progress bar
            # ------------------------------

            progress_bar = st.progress(0)

            for percent_complete in range(0, 101):
                time.sleep(0.01)
                progress_bar.progress(percent_complete)

            # ------------------------------
            # Run prediction
            # ------------------------------

            with st.spinner("Analyzing data..."):

                data, suspicious_data = predict_from_csv(
                    temp_path
                )

            filtered_data = suspicious_data

            # ------------------------------
            # End timer
            # ------------------------------

            end_time = time.time()

            processing_time = end_time - start_time

            st.success("Detection completed successfully")

            st.info(
                f"Processing time: {processing_time:.2f} seconds"
            )

            st.divider()

            # ============================================================
            # DATASET SUMMARY
            # ============================================================

            # Dataset metrics

            total_records = len(data)

            suspicious_count = len(suspicious_data)

            high_count = 0

            if "inspection_priority" in suspicious_data.columns:

                high_count = (
                    suspicious_data[
                        "inspection_priority"
                    ] == "HIGH"
                ).sum()

            colA , colC = st.columns(2)

            colA.metric(
                "Records Processed",
                total_records
            )

            

            colC.metric(
                "High Risk Detected",
                high_count
            )

            # ============================================================
            # KPI METRICS
            # ============================================================

            total_loss = 0

            if "estimated_loss" in filtered_data.columns:

                total_loss = filtered_data[
                    "estimated_loss"
                ].sum()

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Suspicious Consumers",
                len(filtered_data)
            )

            col2.metric(
                "Estimated Revenue Loss",
                f"₹{total_loss:,.0f}"
            )

            st.divider()

            # ============================================================
            # FILTER
            # ============================================================

            if "inspection_priority" in filtered_data.columns:

                priority_filter = st.selectbox(
                    "Filter by Risk Level",
                    ["All", "HIGH", "MEDIUM", "LOW"]
                )

                if priority_filter != "All":

                    filtered_data = filtered_data[
                        filtered_data[
                            "inspection_priority"
                        ] == priority_filter
                    ]

            st.divider()

            # ============================================================
            # DATA TABLE
            # ============================================================

            st.subheader("Possible Theft Locations")

            columns_to_show = [
                col for col in [
                    "risk_score",
                    "inspection_priority",
                    "estimated_loss",
                    "reason",
                    "confidence"
                ] if col in filtered_data.columns
            ]

            st.dataframe(
                filtered_data[columns_to_show],
                use_container_width=True
            )

            st.divider()

            # ============================================================
            # TOP 10 HIGH-RISK CONSUMERS
            # ============================================================

            if "risk_score" in filtered_data.columns:

                st.subheader("Top 10 High-Risk Consumers")

                top_risk = filtered_data.sort_values(
                    by="risk_score",
                    ascending=False
                ).head(10)

                st.dataframe(
                    top_risk,
                    use_container_width=True
                )

            st.divider()

            # ============================================================
            # CHARTS
            # ============================================================

            colX, colY = st.columns(2)

            if "inspection_priority" in filtered_data.columns:

                with colX:

                    st.subheader("Risk Distribution")

                    counts = filtered_data[
                        "inspection_priority"
                    ].value_counts()

                    fig, ax = plt.subplots()

                    counts.plot(
                        kind="bar",
                        ax=ax
                    )

                    st.pyplot(fig)

            if "reason" in filtered_data.columns:

                with colY:

                    st.subheader("Detection Reasons")

                    reason_counts = filtered_data[
                        "reason"
                    ].value_counts()

                    fig2, ax2 = plt.subplots()

                    reason_counts.plot(
                        kind="bar",
                        ax=ax2
                    )

                    st.pyplot(fig2)

            st.divider()

            # ============================================================
            # CONFIDENCE HISTOGRAM
            # ============================================================

            if "confidence" in filtered_data.columns:

                st.subheader("Model Confidence Distribution")

                fig3, ax3 = plt.subplots()

                ax3.hist(
                    filtered_data[
                        "confidence"
                    ],
                    bins=20
                )

                st.pyplot(fig3)

            st.divider()

            # ============================================================
            # TREND VISUALIZATION (FIXED)
            # ============================================================

            if len(filtered_data) > 0:

                st.subheader("Consumption Trend Visualization")

                sample_row = filtered_data.iloc[0]

                numeric_values = pd.to_numeric(
                    sample_row,
                    errors="coerce"
                ).dropna()

                if len(numeric_values) > 0:

                    st.line_chart(numeric_values)

            st.divider()

            # ============================================================
            # DOWNLOAD REPORT
            # ============================================================

            output_file = "possible_theft_locations.csv"

            filtered_data.to_csv(
                output_file,
                index=False
            )

            with open(
                output_file,
                "rb"
            ) as f:

                st.download_button(
                    label="Download Full Report",
                    data=f,
                    file_name=output_file,
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Error occurred: {str(e)}"
        )

    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)