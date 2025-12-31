import streamlit as st

REQUIRED_KEYS = {
    "inputs": None,
    
}

def init_session_state():
    for key, default in REQUIRED_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default
        