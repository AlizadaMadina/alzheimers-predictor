import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def show_results():
    st.title("🧠 Prediction Results")
    st.markdown("---")

    # Load model
    with open("../model/alzheimers_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Get input data from session state
    input_scaled = st.session_state.input_scaled
    input_data = st.session_state.input_data

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Map prediction to label
    labels = {0: "Nondemented", 1: "Demented", 2: "Converted"}
    colors = {0: "green", 1: "red", 2: "orange"}
    prediction_label = labels[prediction]
    prediction_color = colors[prediction]

    # Show prediction
    st.header("Diagnosis Prediction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background-color: {prediction_color}; 
                    padding: 30px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white;'>{prediction_label}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Confidence Scores")
        for i, label in labels.items():
            prob = probabilities[i] * 100
            st.metric(label=label, value=f"{prob:.1f}%")

    st.markdown("---")

    # Feature importance chart
    st.header("What influenced this prediction?")

    feature_names = input_data.columns.tolist()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color="steelblue")
    ax.set_title("Feature Importance for This Prediction")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

    st.markdown("---")

    # Patient vs dataset comparison
    st.header("How does this patient compare to the dataset?")

    df_orig = pd.read_csv("../data/oasis_longitudinal.csv")
    
    col3, col4, col5 = st.columns(3)

    with col3:
        avg_mmse = df_orig.groupby("Group")["MMSE"].mean()
        patient_mmse = input_data["MMSE"].values[0]
        st.metric(label="Patient MMSE Score",
                 value=f"{patient_mmse:.1f}",
                 delta=f"{patient_mmse - df_orig['MMSE'].mean():.1f} vs dataset avg")

    with col4:
        patient_nwbv = input_data["nWBV"].values[0]
        st.metric(label="Patient Brain Volume",
                 value=f"{patient_nwbv:.3f}",
                 delta=f"{patient_nwbv - df_orig['nWBV'].mean():.3f} vs dataset avg")

    with col5:
        patient_age = input_data["Age"].values[0]
        st.metric(label="Patient Age",
                 value=f"{patient_age:.0f}",
                 delta=f"{patient_age - df_orig['Age'].mean():.1f} vs dataset avg")

    st.markdown("---")

    # Navigation buttons
    col6, col7 = st.columns(2)
    with col6:
        if st.button("Make Another Prediction", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
    with col7:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()