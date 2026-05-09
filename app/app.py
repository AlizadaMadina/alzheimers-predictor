import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Predictor",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for navigation and results
if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# Navigation
if st.session_state.page == "home":
    from home import show_home
    show_home()

elif st.session_state.page == "form":
    from form import show_form
    show_form()

elif st.session_state.page == "results":
    from results import show_results
    show_results()