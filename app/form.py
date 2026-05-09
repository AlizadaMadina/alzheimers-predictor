import streamlit as st
import pandas as pd
import pickle

def show_form():
    st.title("🧠 Patient Information Form")
    st.write("Please fill in the patient's clinical measurements below.")
    st.markdown("---")

    # Load scaler
    with open("../model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load feature names
    feature_names = pd.read_csv("../data/X_scaled.csv").columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", min_value=60, max_value=98, value=75)
        gender = st.selectbox("Gender", ["Female", "Male"])
        educ = st.slider("Years of Education", min_value=1, max_value=23, value=12)
        ses = st.slider("Socioeconomic Status (1=High, 5=Low)",
                       min_value=1, max_value=5, value=2)
        visit = st.slider("Visit Number", min_value=1, max_value=5, value=1)

    with col2:
        st.subheader("Clinical Measurements")
        mmse = st.slider("Current MMSE Score (0=Severe, 30=Normal)",
                        min_value=0, max_value=30, value=27)
        etiv = st.slider("eTIV (Total Brain Volume)",
                        min_value=1100, max_value=2000, value=1450)
        nwbv = st.slider("nWBV (Normalized Brain Volume)",
                        min_value=0.64, max_value=0.90,
                        value=0.73, step=0.01)
        asf = st.slider("ASF (Atlas Scaling Factor)",
                       min_value=0.88, max_value=1.59,
                       value=1.19, step=0.01)

    st.markdown("---")

    # Only show first visit fields if visit > 1
    if visit > 1:
        st.subheader("Baseline Measurements from Visit 1")
        st.write("""
        Please enter the patient's MMSE score and brain volume 
        from their very first visit. We always compare current 
        measurements to Visit 1 to calculate the total amount 
        of change since the patient first came in.
        """)
        
        col3, col4 = st.columns(2)
        with col3:
            first_mmse = st.slider("MMSE Score at First Visit",
                                  min_value=0, max_value=30, value=27)
        with col4:
            first_nwbv = st.slider("Brain Volume (nWBV) at First Visit",
                                  min_value=0.64, max_value=0.90,
                                  value=nwbv, step=0.01)
        
        # Calculate engineered features automatically
        mmse_decline = first_mmse - mmse
        nwbv_change = first_nwbv - nwbv
        age_first_visit = age - (visit - 1)
    else:
        # First visit -- no decline yet
        mmse_decline = 0.0
        nwbv_change = 0.0
        age_first_visit = age

    st.markdown("---")

    if st.button("Get Prediction", type="primary", use_container_width=True):
        gender_encoded = 1 if gender == "Male" else 0

        input_data = pd.DataFrame([[
            visit, gender_encoded, age, educ, ses, mmse,
            etiv, nwbv, asf, mmse_decline, nwbv_change, age_first_visit
        ]], columns=feature_names)

        input_scaled = scaler.transform(input_data)

        st.session_state.input_data = input_data
        st.session_state.input_scaled = input_scaled
        st.session_state.page = "results"
        st.rerun()

    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()