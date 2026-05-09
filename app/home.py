import streamlit as st

def show_home():
    # Header section
    st.title("🧠 Alzheimer's Disease Predictor")
    st.subheader("Early detection saves lives.")
    
    st.markdown("---")
    
    # What is this app
    st.header("What is this app?")
    st.write("""
    This web application uses a machine learning model trained on real 
    clinical data to predict whether a patient is likely to be 
    Nondemented, Demented, or in a Converted state based on their 
    cognitive scores and brain measurements.
    
    The model was trained on the OASIS Longitudinal dataset which 
    contains data from 150 patients aged 60 to 98, collected over 
    several years at Washington University.
    """)
    
    st.markdown("---")
    
    # Why early detection matters
    st.header("Why does early detection matter?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="People affected worldwide", value="55 Million")
        st.write("Alzheimer's is the most common cause of dementia globally.")
    
    with col2:
        st.metric(label="New cases every year", value="10 Million")
        st.write("Early detection can slow progression and improve quality of life.")
    
    with col3:
        st.metric(label="Years before symptoms", value="20 Years")
        st.write("Brain changes begin up to 20 years before symptoms appear.")
    
    st.markdown("---")
    
    # How it works
    st.header("How does it work?")
    st.write("""
    1. You fill in a simple form with the patient's clinical measurements
    2. Our Random Forest model analyzes the inputs
    3. You receive a prediction with a confidence percentage
    4. You see which factors influenced the prediction most
    """)
    
    st.markdown("---")
    
    # Call to action button
    st.header("Ready to get a prediction?")
    st.write("Fill in the patient form and get a result in seconds.")
    
    if st.button("Go to Patient Form", type="primary", use_container_width=True):
        st.session_state.page = "form"
        st.rerun()