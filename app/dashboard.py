import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import shap

st.set_page_config(layout="wide")

# ---------------------------------------------------
# FIXED PATH HANDLING
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

final_df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "final_with_interventions.csv"))
engineered_df = pd.read_csv(os.path.join(BASE_DIR, "data", "engineered_behaviour_features.csv"))
raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "synthetic_student_burnout_data_2000.csv"))

# 🔥 Load precomputed SHAP
shap_df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "shap_student_level_values.csv"))
global_shap_df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "shap_global_importance.csv"))

profile_map = raw_df[["student_id", "profile_type"]].drop_duplicates()
final_df = final_df.merge(profile_map, on="student_id", how="left")

# Risk segmentation
def segment(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

final_df["risk_segment"] = final_df["risk_score"].apply(segment)

# ---------------------------------------------------
# PAGE NAVIGATION
# ---------------------------------------------------
page = st.sidebar.radio("Navigation", ["📊 System Overview", "👤 Individual Student Analysis"])

# ===================================================
# PAGE 1: SYSTEM OVERVIEW
# ===================================================
if page == "📊 System Overview":

    st.title("🎓 Behavioural Early Warning System - Overview")

    # =========================
    # KPI METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(final_df))
    col2.metric("High Risk Students", len(final_df[final_df["risk_segment"] == "High"]))
    col3.metric("Average Risk Score", round(final_df["risk_score"].mean(), 2))

    st.markdown("---")

    # =========================
    # ROW 1: Risk Segmentation + Histogram
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Risk Segmentation")

        risk_colors = {
            "Low": "#2ECC71",      # Green
            "Medium": "#F1C40F",   # Yellow
            "High": "#E74C3C"      # Red
        }

        donut = px.pie(
            final_df,
            names="risk_segment",
            hole=0.55,
            color="risk_segment",
            color_discrete_map=risk_colors,
            title="Risk Category Distribution"
        )

        donut.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(donut, use_container_width=True)

    with colB:
        st.subheader("📈 Risk Score Distribution")

        hist = px.histogram(
            final_df,
            x="risk_score",
            nbins=30,
            color_discrete_sequence=["#3498DB"],  # Clean blue
            title="Risk Score Distribution"
        )

        hist.update_traces(marker_line_width=0)
        st.plotly_chart(hist, use_container_width=True)

    st.markdown("---")

    # =========================
    # ROW 2: Profile Risk + SHAP
    # =========================
    colC, colD = st.columns(2)

    with colC:
        st.subheader("📘 Average Risk by Behaviour Profile")

        profile_risk = final_df.groupby("profile_type")["risk_score"].mean().reset_index()

        profile_colors = {
            "consistent": "#1ABC9C",
            "gradual_burnout": "#F39C12",
            "sudden_disengagement": "#E74C3C",
            "chronic_low": "#9B59B6"
        }

        bar_profile = px.bar(
            profile_risk,
            x="profile_type",
            y="risk_score",
            color="profile_type",
            color_discrete_map=profile_colors,
            title="Risk Score by Behaviour Profile"
        )

        st.plotly_chart(bar_profile, use_container_width=True)

    with colD:
        st.subheader("🔬 Global Behavioural Risk Drivers")

        shap_bar = px.bar(
            global_shap_df.head(10),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="purples",
            title="Top Behavioural Risk Drivers"
        )

        shap_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(shap_bar, use_container_width=True)

    st.markdown("---")

    # =========================
    # TOP 10 HIGH RISK TABLE
    # =========================
    st.subheader("🚨 Top 10 High Risk Students")

    top10 = final_df.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(
        top10[["student_id", "risk_score", "burnout_prediction", "dropout_probability"]],
        use_container_width=True
    )

    st.markdown("---")

    # =========================
    # TREND FOR HIGHEST RISK STUDENT
    # =========================
    st.subheader("📈 Behaviour Trend of Highest Risk Student")

    top_student_id = top10.iloc[0]["student_id"]
    trend_df = raw_df[raw_df["student_id"] == top_student_id]

    trend_chart = px.line(
        trend_df,
        x="week",
        y=["attendance", "lms_logins", "sentiment_score"],
        color_discrete_sequence=["#2ECC71", "#3498DB", "#E74C3C"],
        title=f"Behaviour Trends - Student {top_student_id}"
    )
    st.plotly_chart(trend_chart, use_container_width=True)


# ===================================================
# PAGE 2: INDIVIDUAL STUDENT ANALYSIS
# ===================================================
if page == "👤 Individual Student Analysis":

    st.title("👤 Individual Student Behaviour Analysis")

    student_id = st.sidebar.selectbox(
        "Select Student ID",
        final_df["student_id"].unique()
    )

    student_row = final_df[final_df["student_id"] == student_id].iloc[0]
    student_trend = raw_df[raw_df["student_id"] == student_id]

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", student_row["risk_score"])
    col2.metric("Burnout Level", student_row["burnout_prediction"])
    col3.metric("Dropout Probability", student_row["dropout_probability"])

    st.markdown("### 🩺 Recommended Intervention")
    st.success(student_row["recommended_intervention"])

    st.markdown("### 📘 Behaviour Profile")
    st.info(student_row["profile_type"])

    st.markdown("---")

    # 🔬 SHAP Waterfall (FAST — from precomputed CSV)
    st.subheader("🔬 SHAP Explanation for This Student")

    student_shap = shap_df[shap_df["student_id"] == student_id]
    shap_values = student_shap.drop("student_id", axis=1).values[0]

    feature_values = engineered_df[
        engineered_df["student_id"] == student_id
    ].drop(["student_id", "burnout_label", "dropout_label"], axis=1).values[0]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=0,
        data=feature_values,
        feature_names=engineered_df.drop(
            ["student_id", "burnout_label", "dropout_label"], axis=1
        ).columns
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)

    st.markdown("---")

    # Mood trend
    st.subheader("😊 Sentiment (Mood) Trend Over Weeks")
    mood_fig = px.line(
        student_trend,
        x="week",
        y="sentiment_score",
        title="Mood Trend"
    )
    st.plotly_chart(mood_fig, use_container_width=True)

    # Full behavioural trends
    st.subheader("📈 Full Behavioural Trends")
    trend_fig = px.line(
        student_trend,
        x="week",
        y=["attendance", "lms_logins", "submission_delay"],
        title="Weekly Behaviour Trends"
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("Behavioural Early Warning System | Explainable AI Enabled")