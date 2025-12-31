import streamlit as st
import pandas as pd
import time 
from state.session import init_session_state
# Stock list getting

init_session_state()

STOCKLIST = pd.read_csv("risk_dashboard\\data\\Stock List.csv", header=0)



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
        
        weights = [float(i) for i in weights if i.strip()]
        
        if len(weights) != len(tickers):
            st.error("Number of Weights must be the same as the Number of Tickers")
        
        if sum(weights) != 1:
            st.error("Weights must sum to 1")
        
        portfolio_value = st.number_input("Enter value of portfolio ($):", 
                                        placeholder="100000", min_value=100)
        
        
        
    if portfolio_weighting == "No. Shares":
        st.subheader("Enter numbers of shares owned for each asset:")
        shares = st.text_input("Please ensure number of shares owned are separated by comma: ",
                            placeholder="12, 0.231, 1, 223")

    submitted = st.form_submit_button("Configure")

if submitted: 
    with st.spinner("Configuring..."):
        time.sleep(2)