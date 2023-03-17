import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from neuralprophet import NeuralProphet                                            #Importing prophet (prediction and forecasting library.)
from sklearn.metrics import mean_absolute_percentage_error
import mpld3
import streamlit.components.v1 as components
from st_aggrid import AgGrid

def launch():
    tab1, tab2 = st.tabs(["Neural Prophet", "Prophet"])
    with tab1:
        with st.form("Form_Neural_Prophet"):
            st.subheader("Saham")
            option = st.selectbox(
            'Pilih nama Saham yang ingin diprediksi',
            ('BBCA', 'AALI', 'MEGA', 'BUMI', 'BBRI'))
            number = st.number_input('Jangka waktu Prediksi (Hari)',format="%d",value=0,min_value=0,help="nilai minimal adalah 0 dan input harus berupa bilangan bulat")
            submitted = st.form_submit_button("Submit")
            if submitted:
                predict(option,number,"Neural")

        with tab2:
            with st.form("Form_Prophet"):
                st.subheader("Saham")
                option = st.selectbox(
                'Pilih nama Saham yang ingin diprediksi',
                ('BBCA', 'AALI', 'MEGA', 'BUMI', 'BBRI'))
                number = st.number_input('Jangka waktu Prediksi (Hari)',format="%d",value=0,min_value=0,help="nilai minimal adalah 0 dan input harus berupa bilangan bulat")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    predict(option,number,"Prophet")


    pass

def predict(saham,waktu,model_type):
    df = yf.download(saham+'.JK', start='2017-01-01' , end='2022-01-01') 
    df = df.reset_index()
    css = """
    <style>
        .ag-root-wrapper {
            height: 400px;
        }
        .ag-center-cols-container {
            overflow-y: scroll !important;
        }
    </style>
    """
    # df1 = yf.download(saham+'.JK', start='2017-01-01' , end='2023-01-01') 
    # df1 = df1.reset_index()    
    # df1 = df1.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
    if(model_type == "Neural"):
        df = df.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
        m = "my_app/NP_"+saham+".pkl"
        model = pickle.load(open(m, "rb"))                                         
        # model.fit(df)                                                         
        future = model.make_future_dataframe(df,periods = waktu, n_historic_predictions = True)                            
        forecast = model.predict(future)   
        fig = plt.figure(figsize=(16, 9),dpi=100)
        plt.plot(df['ds'], df['y'], 'y' ,label = "Actual")         
        plt.plot(forecast['ds'], forecast['yhat1'], 'k', label = "Predicted")     
        plt.legend()
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=1000, width= 1600)  
        forecast = forecast[forecast["ds"] >= "2022-01-01"]
        fc_table = forecast.reset_index()
        df_table = fc_table.rename(columns={'ds': 'Date','yhat1':'Close'})[['Date', 'Close']]
    else :
        df = df.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
        model = pickle.load(open("my_app/FBP_"+saham+".pkl", "rb"))                                         
        # model.fit(df)                                                         
        future = model.make_future_dataframe(periods = waktu)                            
        forecast = model.predict(future)   
        fig = plt.figure(figsize=(16, 9),dpi=100)
        plt.plot(df['ds'], df['y'], 'y' ,label = "Actual")         
        plt.plot(forecast['ds'], forecast['yhat'], 'k', label = "Predicted")     
        plt.legend()
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=1000, width= 1600)  
        forecast = forecast[forecast["ds"] >= "2022-01-01"]
        fc_table = forecast.reset_index()
        df_table = fc_table.rename(columns={'ds': 'Date','yhat':'Close'})[['Date', 'Close']]

    with st.container():
        st.markdown(css, unsafe_allow_html=True)
        AgGrid(df_table,height=400)       
                                            



# def exp(saham,waktu):
#     df = yf.download(saham+'.JK', start='2017-01-01' , end='2022-01-01') 
#     df = df.reset_index()
#     df1 = yf.download(saham+'.JK', start='2017-01-01' , end='2023-01-01') 
#     df1 = df1.reset_index()    
#     df1 = df1.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
#     df = df.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
#     model = NeuralProphet()                                             
#     model.fit(df)                                                         
#     future = model.make_future_dataframe(df,periods = waktu)                            
#     forecast = model.predict(future)   
#     fig = plt.figure(figsize=(16, 9),dpi=100)
#     plt.plot(df1['ds'], df1['y'], 'y')         
#     plt.plot(forecast['ds'], forecast['yhat1'], 'k')     
#     fig_html = mpld3.fig_to_html(fig)
#     components.html(fig_html, height=1000, width= 1600)                                                       #plotting the values in forecast using atplotlib.



