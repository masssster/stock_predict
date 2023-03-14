import streamlit as st

def launch():
    with st.container():
        st.header('Stock Predict')
        st.write('Stock Predict is experimental application to do stock price prediction.')
        st.write('Stock Predict will do stock price prediction based on best model from skripsi experiment.')
    with st.container():
        st.header('Neural prophet')
        st.write("NeuralProphet bridges the gap between traditional time-series models and deep learning methods.")
        st.write("It's based on PyTorch and can be installed using pip.")
        st.write('Neural prophet is the best model for stock price prediction based on skripsi experiment.')
        st.write('More information about neural prophet from https://neuralprophet.com/')
    with st.container():
        st.header('Prophet')
        st.write("Prophet is a forecasting procedure implemented in  R and Python.")
        st.write("It is fast and provides completely automated forecasts that can be tuned by hand by data scientists and analysts.")
        st.write('prophet is the good enough model for stock price prediction based on skripsi experiment.')
        st.write('More information about prophet from https://facebook.github.io/prophet/')
        
    pass