import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# Load Model Assets
# --------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("fatality_rf_model.pkl")
    features = joblib.load("model_features.pkl")
    threshold = joblib.load("model_threshold.pkl")
    return model, features, threshold

model, FEATURES, THRESHOLD = load_assets()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🚦 Accident Severity Prediction System")
st.markdown(
    """
    **Research-grade machine learning system** designed to predict  
    **fatal injury risk in road accidents** to support **emergency response planning**.
    """
)

st.info(
    """
    **Model:** Random Forest Classifier  
    **Objective:** Early identification of high-risk accident scenarios  
    **Decision Threshold:** Optimized for recall (life-saving priority)
    """
)

st.divider()

# --------------------------------------------------
# Layout: Inputs | Output
# --------------------------------------------------
left, right = st.columns([1.2, 1])

# =========================
# INPUT PANEL
# =========================
with left:
    st.subheader("📥 Accident Information")

    age = st.slider("Person Age", 0, 100, 35)
    hour = st.slider("Crash Hour (0–23)", 0, 23, 14)

    day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    st.markdown("### ⚠️ Contextual Risk Factors")

    col1, col2 = st.columns(2)

    with col1:
        is_weekend = st.toggle("Weekend")
        is_rush_hour = st.toggle("Rush Hour")
        is_night = st.toggle("Night Time")

    with col2:
        is_pedestrian = st.toggle("Pedestrian Involved")
        is_bicyclist = st.toggle("Bicyclist Involved")

    st.markdown("### 🛑 Safety Conditions")

    col3, col4 = st.columns(2)
    with col3:
        ejection_risk = st.toggle("Ejection Risk")
    with col4:
        no_safety_equipment = st.toggle("No Safety Equipment")

# =========================
# OUTPUT PANEL
# =========================
with right:
    st.subheader("📊 Risk Assessment")

    input_data = {
        "PERSON_AGE": age,
        "CRASH_HOUR": hour,
        "DAY_OF_WEEK": day_map[day],
        "IS_WEEKEND": int(is_weekend),
        "IS_RUSH_HOUR": int(is_rush_hour),
        "IS_NIGHT": int(is_night),
        "IS_PEDESTRIAN": int(is_pedestrian),
        "IS_BICYCLIST": int(is_bicyclist),
        "EJECTION_RISK": int(ejection_risk),
        "NO_SAFETY_EQUIPMENT": int(no_safety_equipment)
    }

    input_df = pd.DataFrame([input_data])[FEATURES]

    if st.button("🔍 Predict Accident Severity", use_container_width=True):

        prob = model.predict_proba(input_df)[0][1]
        risk_percent = prob * 100

        # Probability Gauge (Text-based, professional)
        st.metric(
            label="Estimated Fatality Risk",
            value=f"{risk_percent:.2f} %",
            delta=f"Threshold: {THRESHOLD * 100:.0f}%"
        )

        # Risk Categorization
        if prob >= THRESHOLD:
            st.error("🚨 HIGH RISK: Fatal injury likely")
        elif prob >= 0.25:
            st.warning("⚠️ MODERATE RISK: Elevated injury severity")
        else:
            st.success("✅ LOW RISK: Fatal injury unlikely")

        # Probability Bar
        st.progress(min(int(risk_percent), 100))

# --------------------------------------------------
# Explanation Section
# --------------------------------------------------
st.divider()
st.subheader("🧠 Model Interpretation (User-Friendly)")

st.markdown(
    """
    **How to read the result:**

    - The system estimates **probability**, not certainty  
    - A **lower decision threshold (0.40)** is used intentionally  
    - This prioritizes **detecting severe cases early**, even at the cost of false alarms  

    **Why this matters:**  
    Missing a fatal-risk case is far more costly than flagging a non-fatal one.
    """
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Accident Severity Prediction | Machine Learning Research Project | Rachit Patwa"
)
