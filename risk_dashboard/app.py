import streamlit as st 


page_titles = ["Portfolio Summary", "Realised Risk", "Forecast Risk",
                       "Factor Exposure", "Stress Testing", "Concentration Risk",
                       "Liquidity Risk", "Volatility-Based Sizing", "Reconstructured Price",
                       "Themes & Proxies", "Portfolio Composition"]



pages = []

for title in page_titles:
    page = st.Page("pages\\" + title.replace(" ", "_").lower() + ".py", title=title)
    pages.append(page)
    
pg = st.navigation(pages)
pg.run()