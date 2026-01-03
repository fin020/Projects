import streamlit as st
import pandas as pd
import time 
from risk_dashboard.state.app_session import init_session_state
from pathlib import Path
# Stock list getting

init_session_state()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOCKLIST = pd.read_csv(DATA_DIR / "Stock List.csv", header=0)



st.title("Portfolio Composition")
st.divider()
st.info("Enter all the symbols of stocks in your portfolio and their weightings and the total portfolio value. You can also the number of shares you own in each equity. ")

portfolio_weighting = st.radio("Configuration options:", ["Weight", "No. Shares"])


with st.form("input_form"):
    tickers = st.multiselect("Select Tickers in your portfolio:", STOCKLIST['Symbol'])
    start = st.date_input("Start Date")
    st.divider()
    

    if portfolio_weighting == 'Weight':
        st.subheader("Enter Portfolio Weights:")
        
        weights = st.text_input("Please ensure all weightings are separated by a comma: ", 
                    placeholder="0.2, 0.2, 0.2, 0.2, 0.2")
        
        weights = weights.split(",")
        try:
            weights = [float(i) for i in weights if i.strip()]
        except:
            
        
        if not all(weight > 0 for weight in weights):
            st.error("Weights must all be positive values. ")
        
        if len(weights) != len(tickers):
            st.error("Number of Weights must be the same as the Number of Tickers")
        
        if not abs(sum(weights) - 1) < 1e-6:
            st.error("Weights must sum to 1")
        
        portfolio_value = st.number_input("Enter value of portfolio ($):", 
                                        placeholder="100000", min_value=100)
        
        weight_checks = (all(weight > 0 for weight in weights) and len(weights) == len(tickers)
                and abs(sum(weights) - 1) < 1e-6)
        
        
        
    if portfolio_weighting == "No. Shares":
        st.subheader("Enter numbers of shares owned for each asset:")
        shares = st.text_input("Please ensure number of shares owned are separated by comma: ",
                            placeholder="12, 0.231, 1, 223")
    
        shares = shares.split(",")
        
        shares = [float(share) for share in shares if share.strip()]
        
        if len(shares) != len(tickers):
            st.error("count of shares must be the same as the count of Tickers")

    submitted = st.form_submit_button("Configure")
    
    if portfolio_weighting == "Weight":
        if submitted and weight_checks:
            st.spinner("configuring...")
            st.session_state["inputs"] = {
                "tickers": tickers,
                "weights": weights,
                "shares": shares,
                "portfolio_value": portfolio_value,
                "start_date": start,
                "config_type": portfolio_weighting
            }
            time.sleep(2)
        
    if portfolio_weighting == "No. Shares":
        if submitted and len(shares) == len(tickers):
            st.spinner("configuring...")
            st.session_state["inputs"] = {
                "tickers": tickers,
                "weights": weights,
                "shares": shares,
                "portfolio_value": portfolio_value,
                "start_date": start,
                "config_type": portfolio_weighting
            }
            time.sleep(2)
