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
            st.subheader("Stock")
            option = st.selectbox(
            'Choose which stock to predict',
            ('BBCA', 'AALI', 'MEGA', 'BUMI', 'BBRI'))
            number = st.number_input('Prediction Timeframe (Day)',format="%d",value=0,min_value=0,max_value = 9999,help="min value is 0, max value is 9999, and input must be integer")
            submitted = st.form_submit_button("Predict")
            if submitted:
                predict(option,number,"Neural")

        with tab2:
            with st.form("Form_Prophet"):
                st.subheader("Stock")
                option = st.selectbox(
                'Choose which stock to predict',
                ('BBCA', 'AALI', 'MEGA', 'BUMI', 'BBRI'))
                number = st.number_input('Prediction Timeframe (Day)',format="%d",value=0,min_value=0,max_value = 9999,help="min value is 0, max value is 9999, and input must be integer")
                submitted = st.form_submit_button("Predict")
                if submitted:
                    predict(option,number,"Prophet")


    pass

def predict(saham,waktu,model_type):
    param_neural = {"BBCA" : [100,"multiplicative",0.1,0.001],
                    "AALI" : [100,"additive",10,100],
                    "MEGA" : [150,"additive",0.1,0.01],
                    "BUMI" : [200,"multiplicative",100,1],
                    "BBRI" : [150,"multiplicative",100,0.1],
    }
    param_prophet = {"BBCA" : [0.001, 0.1, 200, "multiplicative"],
                    "AALI" : [0.01, 0.01, 200, "multiplicative"],
                    "MEGA" : [0.1, 0.01, 100, "multiplicative"],
                    "BUMI" : [0.1, 0.3, 150, "additive"],
                    "BBRI" : [0.5, 0.01, 100, "multiplicative"],
    }
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
    with st.container():
        st.header('Dataset')
        st.write("Data : " + saham + ".")
        st.write("Timeframe : 2017-01-01 until 2022-01-01.")
        st.write('Target Column : Close')

    with st.container():
        if(model_type == "Neural"):
            st.header('Modeling Parameter')
            st.write("n_changepoints : " + str(param_neural[saham][0]))
            st.write("seasonality_mode : " + str(param_neural[saham][1]))
            st.write('seasonality_reg : ' + str(param_neural[saham][2]))
            st.write("trend_reg : " + str(param_neural[saham][3]))
            st.write("daily_seasonality : False.")
            st.write('weekly_seasonality : True.')
            st.write("yearly_seasonality : True.")
        if(model_type == "Prophet"):
            st.header('Modeling Parameter')
            st.write("changepoint_prior_scale : " + str(param_prophet[saham][0]))
            st.write("seasonality_prior_scale : " + str(param_prophet[saham][1]))
            st.write('n_changepoints : ' + str(param_prophet[saham][2]))
            st.write("seasonality_mode : " + str(param_prophet[saham][3]))
            st.write("daily_seasonality : False.")
            st.write('weekly_seasonality : True.')
            st.write("yearly_seasonality : True.")

    with st.container():
        st.header('Forecast')
        if(model_type == "Neural"):
            df = df.rename(columns={'Date': 'ds','Close':'y'})[['ds', 'y']]
            m = "my_app/NP_"+saham+".pkl"
            model = pickle.load(open(m, "rb"))                                         
            # model.fit(df)                                                         
            future = model.make_future_dataframe(df,periods = waktu, n_historic_predictions = True)                            
            forecast = model.predict(future)   
            st.write("MAPE : " + str(mean_absolute_percentage_error(forecast['yhat1'][forecast["ds"]  < "2021-12-30"],df['y'])))
            fig = plt.figure(figsize=(16, 9),dpi=100)
            plt.plot(df['ds'], df['y'], 'y' ,label = "Actual")         
            plt.plot(df['ds'], forecast['yhat1'][forecast["ds"] <= "2021-12-30"], 'k', label = "Predicted")     
            plt.plot(forecast['ds'][forecast["ds"] > "2021-12-30"], forecast['yhat1'][forecast["ds"] > "2021-12-30"], 'r', label = "Forecast")     
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
            st.write("MAPE : " + str(mean_absolute_percentage_error(forecast['yhat'][forecast["ds"]  <= "2021-12-30"],df['y']))) 
            fig = plt.figure(figsize=(16, 9),dpi=100)
            plt.plot(df['ds'], df['y'], 'y' ,label = "Actual")         
            plt.plot(df['ds'], forecast['yhat'][forecast["ds"]  <= "2021-12-30"], 'k', label = "Predicted")    
            plt.plot(forecast['ds'][forecast["ds"] > "2021-12-30"], forecast['yhat'][forecast["ds"] > "2021-12-30"], 'r', label = "Forecast")     
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



