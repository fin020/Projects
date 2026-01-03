import streamlit as st

REQUIRED_KEYS = {
    "tickers": list,
    "weights": None,
    "shares": None,
    "portfolio_value": None,
    "start_date": None,
    "config_mode": None,
    "configured": False
    
}

def init_session_state():
    for key, default in REQUIRED_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default
        